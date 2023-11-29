
---
20231128~20231129 K-DS-Hackaton 
---
![image](https://github.com/daily-kim/K-DS-Hackaton_History_LLM/assets/90249131/2887beb0-cc2b-445e-9c78-c8aa265168ee)


* LLM serving
  * Local server (1* RTX 6000)
  * Docker
  * HuggingFace pipeline

* LLM Finetuning
  * Original model from [KoAlpaca-Polyglot-5.8B](https://huggingface.co/beomi/KoAlpaca-Polyglot-5.8B)
  * Finetuned using Peft+LoRA

* [RAG](https://github.com/venzino-han/history_rag)
  * ChromaDB
  * FastAPI
  * Docker

* Gradio
  * Original code from [Baize](https://huggingface.co/spaces/project-baize/chat-with-baize)
