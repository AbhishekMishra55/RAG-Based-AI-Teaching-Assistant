import json

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("Loading processed chunks...")

with open("processed_chunks.json", "r", encoding="utf-8") as file:
    chunk_data = json.load(file)

documents = []

for item in chunk_data:
    documents.append(
        Document(
            page_content=item["content"],
            metadata=item["metadata"]
        )
    )

print("Documents Loaded:", len(documents))

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating FAISS index...")

vector_store = FAISS.from_documents(
    documents,
    embeddings
)

print("Saving FAISS index...")

vector_store.save_local("faiss_index")

print("FAISS Index Saved Successfully!")