
base_model: TheBloke/Llama-2-7B-Chat-GGML
pipeline_tag: question-answering
library_name: transformers
tags:
- medical
- conversational
- text-generation
---
# 🐍 Llama-2-GGML-Medical-Chatbot 🤖
The **Llama-2-7B-Chat-GGML-Medical-Chatbot** is a repository for a medical chatbot that uses the _Llama-2-7B-Chat-GGML_ model and the pdf _The Gale Encyclopedia of Medicine_. The chatbot is still under development, but it has the potential to be a valuable tool for patients, healthcare professionals, and researchers. The chatbot can be used to answer questions about medical topics, provide summaries of medical articles, and generate medical text. 
## 📚 Here are some of the features of the Llama-2-7B-Chat-GGML-Medical-Chatbot:

 - It uses the _Llama-2-7B-Chat-GGML_ model, which is a **large language model (LLM)** that has been fine-tuned.
   * Name - **llama-2-7b-chat.ggmlv3.q2_K.bin**
   * Quant method - q2_K
   * Bits - 2
   * Size - **2.87 GB**
   * Max RAM required - 5.37 GB
   * Use case - New k-quant method. Uses GGML_TYPE_Q4_K for the attention.vw and feed_forward.w2 tensors, GGML_TYPE_Q2_K for the other tensors.
   * **Model:** Know more about model **[Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML)**
 - It is trained on the pdf **[The Gale Encyclopedia of Medicine, Volume 1, 2nd Edition, 637-page PDF]*, which is a comprehensive medical reference that provides information on a wide range of medical topics. This means that the chatbot is able to answer questions about a variety of medical topics.
 - This is a sophisticated medical chatbot, developed using Llama-2 7B and Sentence Transformers. Powered by **[Langchain](https://python.langchain.com/docs/get_started/introduction)** and **[Chainlit](https://docs.chainlit.io/overview)**, This bot operates on a powerful CPU computer that boasts a minimum of
    * Operating system: Linux, macOS, or Windows
    * CPU: Intel® Core™ i3
    * RAM: **8 GB**
    * Disk space: 7 GB
    * GPU: None **(CPU only)**
 - It is still under development, but it has the potential to be a valuable tool for patients, healthcare professionals, and researchers.




