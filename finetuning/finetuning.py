from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def load_and_preprocess_data(file_path):
    data = load_dataset('csv', data_files=file_path)
    return data.map(lambda x: {'text': f"### 질문: {x['Q']}\n\n### 답변: {x['A']} <|Endofsentence|>"})

def initialize_model_and_tokenizer(model_path, tokenizer_path, bnb_config):
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map={"":0})
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def tokenize_data(data, tokenizer):
    return data.map(lambda samples: tokenizer(samples["text"]), batched=True)

def setup_training(model, data, tokenizer):
    trainer = Trainer(
        model=model,
        train_dataset=data["train"],
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=70,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=5,
            output_dir="outputs",
            optim="paged_adamw_8bit"
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    return trainer

def train_model(trainer, model):
    model.config.use_cache = False
    trainer.train()
    model.eval()
    model.config.use_cache = True

def gen(model, tokenizer, x):
    gened = model.generate(
        **tokenizer(
            f"### 질문: {x}\n\n### 답변:", 
            return_tensors='pt', 
            return_token_type_ids=False
        ), 
        max_new_tokens=256,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    return tokenizer.decode(gened[0])


def main():
    # File paths and model paths
    data_file_path = 'k_hist_qa.csv'
    model_path = 'model'
    tokenizer_path = 'tokenizer'
    output_dir = "tuned_model"

    # Load and preprocess data
    data = load_and_preprocess_data(data_file_path)

    # BitsAndBytes Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(model_path, tokenizer_path, bnb_config)

    # Tokenize the data
    tokenized_data = tokenize_data(data, tokenizer)

    # Enable gradient checkpointing and prepare model for kbit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Define LoraConfig and get PEFT model
    config = LoraConfig(r=8, lora_alpha=32, target_modules=["query_key_value"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, config)

    # Setup training
    trainer = setup_training(model, tokenized_data, tokenizer)

    # Train the model
    train_model(trainer, model)

    prompts = ["조선의 초대 왕은 누구인가요", "아관 파천에 대해서 설명하세요.", "임진왜란에 대한 설명을 작성하세요."]
    for prompt in prompts:
        generated_response = gen(model, tokenizer, prompt)
        print(f"Prompt: {prompt}\nGenerated Response: {generated_response}\n")

    # Save the model
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
