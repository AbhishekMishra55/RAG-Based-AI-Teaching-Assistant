from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# ==========================================================
# Load Embedding Model
# ==========================================================

print("Loading Embedding Model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================================
# Load FAISS Vector Store
# ==========================================================

print("Loading FAISS Index...")

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# ==========================================================
# Create Retriever
# ==========================================================

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 30
    }
)

# ==========================================================
# Load Local LLM
# ==========================================================

print("Loading Gemma 3:1B...")

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0
)

print("\n🤖 AI Machine Learning Code Assistant Ready!\n")

# ==========================================================
# Chat Loop
# ==========================================================

while True:

    question = input("Ask Question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # ------------------------------------------------------
    # Retrieve Documents
    # ------------------------------------------------------

    docs = retriever.invoke(question)

    context = ""
    sources = []

    for doc in docs:

        doc_type = doc.metadata.get("type", "unknown")
        source = doc.metadata.get("source", "unknown")

        context += f"""
==========================================================
TYPE   : {doc_type.upper()}
SOURCE : {source}
==========================================================

{doc.page_content}

"""

        if source not in sources:
            sources.append(source)

    # ------------------------------------------------------
    # Debug Retrieved Context
    # ------------------------------------------------------

    print("\nRetrieved Context")
    print("=" * 100)
    print(context[:4000])
    print("=" * 100)

    # ------------------------------------------------------
    # Prompt
    # ------------------------------------------------------

    prompt = f"""
You are an expert Machine Learning tutor and Python coding assistant.

The retrieved context contains both markdown explanations and Python code.

Answer ONLY using the retrieved context.

When possible, organize your answer using this format:

## Concept
Explain the concept briefly.

## Code Explanation
Explain the retrieved code line by line.

## Key Takeaways
Summarize the important points.

If the answer is not present in the retrieved context, reply exactly:

I don't know based on the provided notebook.

========================
CONTEXT
========================

{context}

========================
QUESTION
========================

{question}
"""

    # ------------------------------------------------------
    # Generate Response
    # ------------------------------------------------------

    response = llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
    )

    # ------------------------------------------------------
    # Print Answer
    # ------------------------------------------------------

    print("\n" + "=" * 100)
    print("ANSWER\n")
    print(response.content)

    # ------------------------------------------------------
    # Sources
    # ------------------------------------------------------

    print("\nSources Used:")

    for i, source in enumerate(sources, start=1):
        print(f"{i}. {source}")

    print("=" * 100)
  
        

