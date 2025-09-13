# 🤖 AI FAQ Chatbot with LangChain, Cohere, Gemini & Qdrant

An **interactive FAQ chatbot** powered by **LangChain**, **Google Gemini AI**, **Cohere**, and **Qdrant**.  
It can **answer questions from your documents (PDFs)**, rerank results for better accuracy, and maintain **conversation memory** like a real assistant.

---

## ✨ Features

- 📄 **PDF Ingestion** – Load and process your documents into a vector database.  
- 🔎 **MMR Retriever + MultiQuery Retrieval** – Improves search diversity.  
- 🔄 **Cohere Reranking** – Ensures the most relevant answers.  
- 🧠 **Conversation Memory** – Keeps track of chat history for natural responses.  
- 💬 **Conversational Retrieval Chain** – Chat with your documents seamlessly.  
- ⚡ **Streaming Output** – Real-time AI responses.  
- 🔒 **Environment Variables Support** – Secure API keys via `.env`.

---

## 🛠️ Tech Stack

- [LangChain](https://www.langchain.com/) – Orchestration  
- [Google Gemini](https://ai.google.dev/) – LLM & Embeddings  
- [Cohere](https://cohere.com/) – Reranking  
- [Qdrant](https://qdrant.tech/) – Vector database  
- [Python](https://www.python.org/) – Backend  

---

## 📂 Project Structure

```bash
📦 chatbot-project
┣ 📂 0.0_chat_bot_task
┃ ┣ 📄 data1.pdf
┃ ┣ 📄 data2.pdf
┣ 📄 chatbot.py             # Main chatbot loop
┣ 📄 create_db.py           # Script to load & index documents into Qdrant
┣ 📄 .env                   # Store API keys & configs (not committed to git)
┣ 📄 requirements.txt       # Dependencies
┗ 📄 README.md              # You’re here


## 📂 Project Structure

