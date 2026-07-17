import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

st.set_page_config(
    page_title="RAG-Based Machine Learning Code Assistant",
    page_icon="🤖",
    layout="wide"
)
st.sidebar.title("🤖 AI Assistant")

st.sidebar.markdown("---")

st.sidebar.write("### About")

st.sidebar.info(
    """
This application uses:

✅ LangChain

✅ FAISS Vector Database

✅ HuggingFace Embeddings

✅ Ollama

✅ Gemma 3:1B

to answer Machine Learning questions from notebooks.
"""
)

st.sidebar.markdown("---")

st.sidebar.write("### Technologies")

st.sidebar.write("""
- Python
- Streamlit
- LangChain
- FAISS
- Sentence Transformers
- Ollama
- Gemma 3
""")

st.sidebar.markdown("---")

st.sidebar.success("Ready")


st.write("Loading AI Assistant...")

st.write("Loading Embedding Model...")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

st.success("✅ Embedding Model Loaded")

st.write("Loading FAISS Index...")

@st.cache_resource
def load_vector_store():
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

vector_store = load_vector_store()

st.success("✅ FAISS Index Loaded")

st.write("Loading Gemma 3:1B...")

@st.cache_resource
def load_llm():
    return ChatOllama(
        model="gemma3:1b",
        temperature=0
    )

llm = load_llm()

st.success("✅ Gemma Loaded")

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 30
    }
)

st.title("🤖 RAG-Based Machine Learning Code Assistant")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])
st.write(
    """
Ask any question about Machine Learning concepts or code implementations.
"""
)

question = st.text_input("Ask your Machine Learning question:")
if st.button("🗑 Clear"):
    st.rerun()

if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching Knowledge Base..."):

            docs = retriever.invoke(question)

            context = ""
            sources = []

            for doc in docs:

                doc_type = doc.metadata.get("type", "unknown")
                source = doc.metadata.get("source", "unknown")

                context += f"""
TYPE : {doc_type.upper()}
SOURCE : {source}

{doc.page_content}

"""

                if source not in sources:
                    sources.append(source)

            prompt = f"""
You are an expert Machine Learning tutor and Python coding assistant.

The retrieved context contains markdown explanations and Python code.

Answer ONLY using the context below.

If the answer is not present, reply exactly:

I don't know based on the provided notebook.

Context:

{context}

Question:

{question}
"""

            response = llm.invoke(
                [
                    HumanMessage(content=prompt)
                ]
            )

        st.markdown("---")

        st.subheader("🤖 AI Answer")

        st.markdown(response.content)

        st.markdown("---")

        st.subheader("📚 Sources Used")

        for source in sources:
          with st.expander(source):
           st.write(source)


st.markdown("---")

st.caption(
    "Built by Abhishek Mishra | RAG-Based Machine Learning Code Assistant | LangChain + FAISS + Ollama + Gemma"
)