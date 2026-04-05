# 💹 Finance AI Assistant (Local RAG System)

An AI-powered finance assistant built using a **Retrieval-Augmented Generation (RAG)** pipeline that provides accurate, context-aware answers on personal finance, investing, and economic concepts — running entirely **locally without external APIs**.

---

## 🚀 Features

- 💬 Conversational finance chatbot with context memory  
- 🧠 RAG pipeline using FAISS for semantic retrieval  
- 🔍 Similarity-based search with threshold filtering  
- ⚡ Optional cross-encoder re-ranking for improved accuracy  
- 🤖 Multiple local LLM support (TinyLlama, Phi-2, Phi-3, Flan-T5)  
- 📊 Built-in SIP & savings growth calculators  
- 🔐 100% local inference (no API keys, no data leakage)  

---

## 🏗️ Architecture

User Query

↓

Embedding (Sentence Transformers)

↓

FAISS Vector Search

↓

Top-K Relevant Chunks
↓

(Optional) Re-ranking

↓

Prompt Construction

↓

Local LLM (HF Transformers)

↓

Final Answer

---

## 🖥️ Tech Stack

- Frontend: Streamlit  
- Backend: LangChain (v0.2+)  
- LLMs: HuggingFace Transformers  
- Embeddings: Sentence-Transformers  
- Vector Store: FAISS  
- Language: Python  

---

## 📂 Project Structure

├── app.py
├── rag_pipeline.py
├── embeddings.py
├── llm.py
├── requirements.txt

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/finance-ai-assistant.git
cd finance-ai-assistant
pip install -r requirements.txt
```
▶️ Run

streamlit run app.py

🧠 Key Concepts

Retrieval-Augmented Generation (RAG)
Semantic Search (Dense Embeddings)
Vector Databases (FAISS)
Prompt Engineering
Local LLM Inference

⚠️ Disclaimer

This project is for educational purposes only and does not provide financial advice.

🔮 Future Improvements

Live market data integration
Portfolio analysis & recommendations
PDF/document ingestion
Multi-agent financial planning

⭐ Highlights

Fully local GenAI system
Real-world finance use case
Modular and scalable architecture
No dependency on paid APIs
