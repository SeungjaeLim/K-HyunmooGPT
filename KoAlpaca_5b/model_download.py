
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

MODEL = 'beomi/KoAlpaca-Polyglot-5.8B'
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

model.eval()
# print(model)
#save model to local
model.save_pretrained('KoAlpaca_5.8b/model')
tokenizer.save_pretrained("KoAlpaca_5.8b/tokenizer")
