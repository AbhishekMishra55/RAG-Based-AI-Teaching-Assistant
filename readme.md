# 🤖 RAG-Based Machine Learning Code Assistant

A Retrieval-Augmented Generation (RAG) based AI assistant that answers Machine Learning questions using Jupyter Notebook content. The system retrieves the most relevant notebook chunks using FAISS vector search and generates context-aware responses with Gemma 3 running locally through Ollama.

---

## 📌 Project Overview

This project is designed to help students and beginners learn Machine Learning by asking natural language questions about concepts and code implementations.

Instead of relying on a general-purpose language model, the assistant retrieves relevant notebook content and generates answers grounded in the retrieved context.

The project uses notebooks from *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* as its knowledge base.

---

# 🚀 Features

- Retrieval-Augmented Generation (RAG)
- Semantic Search using FAISS
- Local LLM using Ollama (Gemma 3)
- Sentence Transformer Embeddings
- Interactive Streamlit Interface
- Source Notebook Display
- Code-aware Question Answering
- Context-based Responses

---

# 🛠 Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Sentence Transformers
- Ollama
- Gemma 3
- Jupyter Notebook
- JSON

---

# ⚙️ Project Architecture

```
User Question
      │
      ▼
Streamlit Interface
      │
      ▼
FAISS Retriever
      │
      ▼
Relevant Notebook Chunks
      │
      ▼
Gemma 3 (Ollama)
      │
      ▼
Final Answer
```

---

# 📂 Project Structure

```
RAG-Based-AI-Teaching-Assistant
│
├── app.py
├── rag_pipeline.py
├── chunk_documents.py
├── embeddings.py
├── retriever.py
├── vector_store.py
├── read_notebook.py
├── requirements.txt
│
├── data/
│      ├── Machine Learning notebooks
│
├── faiss_index/
│      ├── index.faiss
│      ├── index.pkl
│
├── screenshots/
│
└── processed_chunks.json
```

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/AbhishekMishra55/RAG-Based-AI-Teaching-Assistant.git
```

Move into the project folder

```bash
cd RAG-Based-AI-Teaching-Assistant
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Install Ollama

https://ollama.com

Download the model

```bash
ollama pull gemma3:1b
```

Start Ollama

```bash
ollama serve
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# 🎯 Usage

Ask questions such as:

- What is Linear Regression?
- Explain Gradient Descent.
- Show Decision Tree implementation.
- Explain Random Forest.
- Show Logistic Regression code.

The assistant retrieves relevant notebook sections and generates answers using the local LLM.

---

# 📸 Screenshots

Add screenshots here.

Example:

- Home Page
- Question Answering
- Retrieved Sources

---

# 🔮 Future Improvements

- Chat History
- PDF Upload Support
- Multiple Knowledge Bases
- Hybrid Search (BM25 + Vector Search)
- Citation Highlighting
- Conversation Memory
- Deployment using Docker

---

# 👨‍💻 Author

**Abhishek Mishra**

B.Tech Computer Science

Aspiring Data Analyst | Data Scientist | Machine Learning Engineer

GitHub:
https://github.com/AbhishekMishra55