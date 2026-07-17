from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading FAISS Index...")

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS Loaded Successfully!")

while True:
    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10}
    )

    results = retriever.invoke(query)

    print("\n" + "=" * 80)

    for i, doc in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("-" * 80)

        print("Source :", doc.metadata.get("source"))
        print("Type   :", doc.metadata.get("type"))

        print("\nContent:\n")
        print(doc.page_content[:800])