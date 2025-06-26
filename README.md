# 🤖 Custom RAG ChatBot with Streamlit & Gemini API

This is a fully customizable chatbot powered by Gemini (Google's LLM) and LangChain's RAG (Retrieval Augmented Generation). Upload your own `.txt`, `.pdf`, or `.docx` files, and ask unlimited questions based on their content.

## 🚀 Features

- 📄 Upload multiple file types: `.txt`, `.pdf`, `.docx`
- 📚 Chunking and embedding using `langchain` and `HuggingFaceEmbeddings`
- 🔍 RAG pipeline with FAISS for semantic search
- 💬 Real-time chat with context-aware answers via Gemini
- 🧠 Session memory: chat history retained during session
- 🔒 Environment variables securely handled (API keys not exposed)

## 🛠️ Tech Stack

- `Streamlit` – UI & deployment
- `LangChain` – Text splitting, RAG, FAISS vectorstore
- `OpenAI` – Gemini-compatible client
- `python-dotenv` – Load API keys securely from `.env`
- `pdfplumber` and `python-docx` – File parsing


