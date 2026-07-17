
import json
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

data_folder = "data"

documents = []

for filename in os.listdir(data_folder):

    if filename.endswith(".ipynb"):

        file_path = os.path.join(data_folder, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            notebook = json.load(file)

        for cell in notebook["cells"]:

            text = "".join(cell["source"]).strip()

            # Skip empty cells
            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "type": cell["cell_type"]
                    }
                )
            )

print("Documents:", len(documents))


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

chunk_data = []

for chunk in chunks:

    chunk_data.append({
        "content": chunk.page_content,
        "metadata": chunk.metadata
    })

with open("processed_chunks.json", "w", encoding="utf-8") as file:

    json.dump(chunk_data, file, indent=4, ensure_ascii=False)

print("Chunks saved successfully!")

print("Chunks:", len(chunks))

print("\nFirst Chunk:\n")
print(chunks[0])
