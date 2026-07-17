from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Model Loaded Successfully!")

text = "Machine Learning is a branch of Artificial Intelligence."

embedding = model.encode(text)

print("Embedding Length:", len(embedding))

print("\nFirst 10 values:")

print(embedding[:10])