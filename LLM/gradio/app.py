# -*- coding:utf-8 -*-
import os
import logging
import sys
import gradio as gr
import torch
import gc
from app_modules.utils import *
from app_modules.presets import *
from app_modules.overwrites import *

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)

# base_model = "project-baize/baize-v2-7b"
# adapter_model = None


# tokenizer,model,device = load_tokenizer_and_model(base_model,adapter_model)
BASE_MODEL_PATH = '/workspace/KoAlpaca_5b/model'
TOKENIZER_PATH = '/workspace/KoAlpaca_5b/tokenizer'
PEFT_PATH = '/workspace/KoAlpaca_5b/model_tuned'

tokenizer,model,device = load_tokenizer_and_model_custom(BASE_MODEL_PATH,TOKENIZER_PATH,PEFT_PATH)

total_count = 0
def predict(text,
            chatbot,
            history,
            top_p,
            temperature,
            max_length_tokens,
            max_context_length_tokens,):
    if text=="":
        yield chatbot,history,"Empty context."
        return 
    try:
        model
    except:
        yield [[text,"No Model Found"]],[],"No Model Found"
        return

    inputs = generate_prompt_with_history(text,history,tokenizer,max_length=max_context_length_tokens)
    if inputs is None:
        yield chatbot,history,"Input too long."
        return 
    else:
        prompt,inputs=inputs
        begin_length = len(prompt)
    input_ids = inputs["input_ids"][:,-max_context_length_tokens:].to(device)
    torch.cuda.empty_cache()
    global total_count
    total_count += 1
    print(total_count)
    if total_count % 50 == 0 :
        os.system("nvidia-smi")
    with torch.no_grad():
        for x in greedy_search(input_ids,model,tokenizer,stop_words=["### 질문:", "### 답변:"],max_length=max_length_tokens,temperature=temperature,top_p=top_p):
            if is_stop_word_or_prefix(x,["### 질문:", "### 답변:"]) is False:
                if "### 질문:" in x:
                    x = x[:x.index("### 질문:")].strip()
                if "### 답변:" in x:
                    x = x[:x.index("### 답변:")].strip() 
                x = x.strip()   
                a, b=   [[y[0],convert_to_markdown(y[1])] for y in history]+[[text, convert_to_markdown(x)]],history + [[text,x]]
                yield a, b, "Generating..."
            if shared_state.interrupted:
                shared_state.recover()
                try:
                    yield a, b, "Stop: Success"
                    return
                except:
                    pass
    del input_ids
    gc.collect()
    torch.cuda.empty_cache()
    print(text)
    print(x)
    print("="*80)
    try:
        yield a,b,"Generate: Success"
    except:
        pass
        
def retry(
        text,
        chatbot,
        history,
        top_p,
        temperature,
        max_length_tokens,
        max_context_length_tokens,
        ):
    logging.info("Retry...")
    if len(history) == 0:
        yield chatbot, history, f"Empty context"
        return
    chatbot.pop()
    inputs = history.pop()[0]
    for x in predict(inputs,chatbot,history,top_p,temperature,max_length_tokens,max_context_length_tokens):
        yield x


gr.Chatbot.postprocess = postprocess

with open("assets/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()

with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
    history = gr.State([])
    user_question = gr.State("")
    with gr.Row():
        gr.HTML(title)
        status_display = gr.Markdown("Success", elem_id="status_display")
    gr.Markdown(description_top)
    with gr.Row(scale=1).style(equal_height=True):
        with gr.Column(scale=5):
            with gr.Row(scale=1):
                chatbot = gr.Chatbot(elem_id="chuanhu_chatbot").style(height="100%")
            with gr.Row(scale=1):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(
                        show_label=False, placeholder="Enter text"
                    ).style(container=False)
                with gr.Column(min_width=70, scale=1):
                    submitBtn = gr.Button("Send")
                with gr.Column(min_width=70, scale=1):
                    cancelBtn = gr.Button("Stop")
            with gr.Row(scale=1):
                emptyBtn = gr.Button(
                    "🧹 New Conversation",
                )
                retryBtn = gr.Button("🔄 Regenerate")
                delLastBtn = gr.Button("🗑️ Remove Last Turn") 
        with gr.Column():
            with gr.Column(min_width=50, scale=1):
                with gr.Tab(label="Parameter Setting"):
                    gr.Markdown("# Parameters")
                    top_p = gr.Slider(
                        minimum=-0,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        interactive=True,
                        label="Top-p",
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.5,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    max_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=512,
                        value=512,
                        step=8,
                        interactive=True,
                        label="Max Generation Tokens",
                    )
                    max_context_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=1024,
                        step=128,
                        interactive=True,
                        label="Max History Tokens",
                    )
    gr.Markdown(description)

    predict_args = dict(
        fn=predict,
        inputs=[
            user_question,
            chatbot,
            history,
            top_p,
            temperature,
            max_length_tokens,
            max_context_length_tokens,
        ],
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )
    retry_args = dict(
        fn=retry,
        inputs=[
            user_input,
            chatbot,
            history,
            top_p,
            temperature,
            max_length_tokens,
            max_context_length_tokens,
        ],
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )

    reset_args = dict(
        fn=reset_textbox, inputs=[], outputs=[user_input, status_display]
    )
 
    # Chatbot
    transfer_input_args = dict(
        fn=transfer_input, inputs=[user_input], outputs=[user_question, user_input, submitBtn], show_progress=True
    )

    predict_event1 = user_input.submit(**transfer_input_args).then(**predict_args)

    predict_event2 = submitBtn.click(**transfer_input_args).then(**predict_args)

    emptyBtn.click(
        reset_state,
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )
    emptyBtn.click(**reset_args)

    predict_event3 = retryBtn.click(**retry_args)

    delLastBtn.click(
        delete_last_conversation,
        [chatbot, history],
        [chatbot, history, status_display],
        show_progress=True,
    )
    cancelBtn.click(
        cancel_outputing, [], [status_display], 
        cancels=[
            predict_event1,predict_event2,predict_event3
        ]
    )    
demo.title = "K-Hyunmoo"

demo.queue(concurrency_count=1).launch()