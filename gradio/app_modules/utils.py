# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Type
import logging
import json
import os
import datetime
import hashlib
import csv
import requests
import re
import html
import markdown2
import torch 
import sys
import gc
from pygments.lexers import guess_lexer, ClassNotFound
import pandas as pd 

import gradio as gr
from pypinyin import lazy_pinyin
import tiktoken
import mdtex2html
from markdown import markdown
from pygments import highlight
from pygments.lexers import guess_lexer,get_lexer_by_name
from pygments.formatters import HtmlFormatter
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM,pipeline, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model
from app_modules.presets import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)

def get_rag(question,n_results=10):
    url = 'http://143.248.90.7:8000/query-kv-top-n'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    data = {
        'query': question,
        'n_results': n_results,
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = pd.DataFrame(response.json()) 
        result = result.sort_values(by=['similarity'], ascending=False)
        context = ' '.join(result['document'].tolist())
        context = context.replace('nan','')
        context = context.replace('\n','')
        context = context.replace("?-?",'')
        # print(context)
    else:
        print(f"Error: {response.status_code}, {response.text}")
        context = ""

    return context

def markdown_to_html_with_syntax_highlight(md_str):
    def replacer(match):
        lang = match.group(1) or "text"
        code = match.group(2)
        lang = lang.strip()
        #print(1,lang)
        if lang=="text":
            lexer = guess_lexer(code)
            lang = lexer.name
            #print(2,lang)
        try:
            lexer = get_lexer_by_name(lang, stripall=True)
        except ValueError:
            lexer = get_lexer_by_name("python", stripall=True)
        formatter = HtmlFormatter()
        #print(3,lexer.name)
        highlighted_code = highlight(code, lexer, formatter)

        return f'<pre><code class="{lang}">{highlighted_code}</code></pre>'

    code_block_pattern = r"```(\w+)?\n([\s\S]+?)\n```"
    md_str = re.sub(code_block_pattern, replacer, md_str, flags=re.MULTILINE)

    html_str = markdown(md_str)
    return html_str


def normalize_markdown(md_text: str) -> str:
    lines = md_text.split("\n")
    normalized_lines = []
    inside_list = False

    for i, line in enumerate(lines):
        if re.match(r"^(\d+\.|-|\*|\+)\s", line.strip()):
            if not inside_list and i > 0 and lines[i - 1].strip() != "":
                normalized_lines.append("")
            inside_list = True
            normalized_lines.append(line)
        elif inside_list and line.strip() == "":
            if i < len(lines) - 1 and not re.match(
                r"^(\d+\.|-|\*|\+)\s", lines[i + 1].strip()
            ):
                normalized_lines.append(line)
            continue
        else:
            inside_list = False
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


def convert_mdtext(md_text):
    code_block_pattern = re.compile(r"```(.*?)(?:```|$)", re.DOTALL)
    inline_code_pattern = re.compile(r"`(.*?)`", re.DOTALL)
    code_blocks = code_block_pattern.findall(md_text)
    non_code_parts = code_block_pattern.split(md_text)[::2]

    result = []
    for non_code, code in zip(non_code_parts, code_blocks + [""]):
        if non_code.strip():
            non_code = normalize_markdown(non_code)
            if inline_code_pattern.search(non_code):
                result.append(markdown(non_code, extensions=["tables"]))
            else:
                result.append(mdtex2html.convert(non_code, extensions=["tables"]))
        if code.strip():
            code = f"\n```{code}\n\n```"
            code = markdown_to_html_with_syntax_highlight(code)
            result.append(code)
    result = "".join(result)
    result += ALREADY_CONVERTED_MARK
    return result

def convert_asis(userinput):
    return f"<p style=\"white-space:pre-wrap;\">{html.escape(userinput)}</p>"+ALREADY_CONVERTED_MARK

def detect_converted_mark(userinput):
    if userinput.endswith(ALREADY_CONVERTED_MARK):
        return True
    else:
        return False



def detect_language(code):
    if code.startswith("\n"):
        first_line = ""
    else:
        first_line = code.strip().split("\n", 1)[0]
    language = first_line.lower() if first_line else ""
    code_without_language = code[len(first_line) :].lstrip() if first_line else code
    return language, code_without_language

def convert_to_markdown(text):
    text = text.replace("$","&#36;")
    def replace_leading_tabs_and_spaces(line):
        new_line = []
        
        for char in line:
            if char == "\t":
                new_line.append("&#9;")
            elif char == " ":
                new_line.append("&nbsp;")
            else:
                break
        return "".join(new_line) + line[len(new_line):]

    markdown_text = ""
    lines = text.split("\n")
    in_code_block = False

    for line in lines:
        if in_code_block is False and line.startswith("```"):
            in_code_block = True
            markdown_text += f"{line}\n"
        elif in_code_block is True and line.startswith("```"):
            in_code_block = False
            markdown_text += f"{line}\n"
        elif in_code_block:
            markdown_text += f"{line}\n"
        else:
            line = replace_leading_tabs_and_spaces(line)
            line = re.sub(r"^(#)", r"\\\1", line)
            markdown_text += f"{line}  \n"

    return markdown_text

def add_language_tag(text):
    def detect_language(code_block):
        try:
            lexer = guess_lexer(code_block)
            return lexer.name.lower()
        except ClassNotFound:
            return ""

    code_block_pattern = re.compile(r"(```)(\w*\n[^`]+```)", re.MULTILINE)

    def replacement(match):
        code_block = match.group(2)
        if match.group(2).startswith("\n"):
            language = detect_language(code_block)
            if language:
                return f"```{language}{code_block}```"
            else:
                return f"```\n{code_block}```"
        else:
            return match.group(1) + code_block + "```"

    text2 = code_block_pattern.sub(replacement, text)
    return text2

def delete_last_conversation(chatbot, history):
    if len(chatbot) > 0:
        chatbot.pop()

    if len(history) > 0:
        history.pop()
        
    return (
        chatbot,
        history,
        "Delete Done",
    )

def reset_state():
    return [], [], "Reset Done"

def reset_textbox():
    return gr.update(value=""),""

def cancel_outputing():
    return "Stop Done"

def transfer_input(inputs):
    # 一次性返回，降低延迟
    textbox = reset_textbox()
    return (
        inputs,
        gr.update(value=""),
        gr.Button.update(visible=True),
    )


class State:
    interrupted = False

    def interrupt(self):
        self.interrupted = True

    def recover(self):
        self.interrupted = False
shared_state = State()





# Greedy Search
def greedy_search(input_ids: torch.Tensor,
                  model: torch.nn.Module,
                  tokenizer: transformers.PreTrainedTokenizer,
                  stop_words: list,
                  max_length: int,
                  temperature: float = 1.0,
                  top_p: float = 1.0,
                  top_k: int = 25) -> Iterator[str]:
    generated_tokens = []
    past_key_values = None
    current_length = 1
    for i in range(max_length):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # apply temperature
            logits /= temperature
    
            probs = torch.softmax(logits, dim=-1)
            # apply top_p
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
    
            # apply top_k
            #if top_k is not None:
            #    probs_sort1, _ = torch.topk(probs_sort, top_k)
            #    min_top_probs_sort = torch.min(probs_sort1, dim=-1, keepdim=True).values
            #    probs_sort = torch.where(probs_sort < min_top_probs_sort, torch.full_like(probs_sort, float(0.0)), probs_sort)
    
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token)
    
            input_ids = torch.cat((input_ids, next_token), dim=-1)
    
            generated_tokens.append(next_token[0].item())
            text = tokenizer.decode(generated_tokens)
    
            yield text
            if any([x in text for x in stop_words]):
                del past_key_values
                del logits
                del probs
                del probs_sort
                del probs_idx
                del probs_sum
                gc.collect()
                return 

def generate_prompt_with_history(text,history,tokenizer,max_length=512):
    prompt = "질문에 대한 답변을 합니다. 답변의 내용은 자세할수록 좋으나, 숫자나 고유명사등은 앞으로 나올 맥락에 나온 것을 따라주세요."   

    history = ["\n### 질문: {} \n### 답변: {}".format(x[0],x[1]) for x in history]
    rag_output = get_rag(text,n_results=3)
    history.append("\n### 맥락: {} ".format(rag_output))
    history.append("\n### 질문: {} \n### 답변: ".format(text))
    # print(history)
    print(rag_output)
    history_text = ""
    flag = False
    for x in history[::-1]:
        if tokenizer(prompt+history_text+x, return_tensors="pt")['input_ids'].size(-1) <= max_length:
            history_text = x + history_text
            flag = True
        else:
            break
    if flag:
        return  prompt+history_text,tokenizer(prompt+history_text, return_tensors="pt")
    else:
        return None


def is_stop_word_or_prefix(s: str, stop_words: list) -> bool:
    for stop_word in stop_words:
        if s.endswith(stop_word):
            return True
        for i in range(1, len(stop_word)):
            if s.endswith(stop_word[:i]):
                return True
    return False



def load_tokenizer_and_model_custom(basemodel_path, tokenizer_path,adapter_model_path):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    peft_config = PeftConfig.from_pretrained(adapter_model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model_base = AutoModelForCausalLM.from_pretrained(
        basemodel_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device=device, non_blocking=True)
    model_base.eval()
    model_to_merge = PeftModel.from_pretrained(model_base,adapter_model_path)
    merged_model = model_to_merge.merge_and_unload()
    
    merged_model.eval()
    del model_to_merge
    del model_base
    gc.collect()
    # merged_model = model_base

    return tokenizer,merged_model,device

