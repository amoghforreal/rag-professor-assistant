# =========================================================
# RAG Academic Assistant Demo
# =========================================================

from app.embeddings.chunker import chunk_text
from app.embeddings.embedder import embed_texts
from app.vectorstore.store import SimpleVectorStore
from app.rag.pipeline import SimpleRAG

import textwrap


# =========================================================
# Demo Study Materials (Simulating Uploaded Files)
# =========================================================

study_materials = {
    "Version Control Notes": """
Git is a distributed version control system used to track changes in source code.
It allows multiple developers to collaborate efficiently.
Git maintains project history and enables branching and merging.
""",

    "Docker Notes": """
Docker is a containerization platform used to package applications along with dependencies.
Containers ensure portability and consistent deployment across environments.
Docker uses images to create containers.
""",

    "RAG AI Notes": """
Retrieval Augmented Generation (RAG) improves AI response quality.
It retrieves relevant knowledge from stored documents before generating answers.
RAG reduces hallucination and improves factual correctness.
"""
}


# =========================================================
# Pretty Printing Helper
# =========================================================

def print_section(title):
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)


def pretty_print(text):
    print(textwrap.fill(text, width=80))


# =========================================================
# Step 1 – Load Documents
# =========================================================

print_section("STEP 1 – Loading Study Materials")

all_text = ""

for title, content in study_materials.items():
    print(f"Loaded: {title}")
    all_text += content + "\n"

print("\nTotal characters loaded:", len(all_text))


# =========================================================
# Step 2 – Chunking
# =========================================================

print_section("STEP 2 – Chunking Documents")

chunks = chunk_text(all_text)

print(f"Total chunks created: {len(chunks)}")

for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1} Preview:")
    pretty_print(chunk)


# =========================================================
# Step 3 – Generate Embeddings
# =========================================================

print_section("STEP 3 – Generating Embeddings")

embeddings = embed_texts(chunks)

print(f"Total embeddings generated: {len(embeddings)}")
print("Example embedding:", embeddings[0])


# =========================================================
# Step 4 – Store in Vector Database
# =========================================================

print_section("STEP 4 – Creating Vector Store")

store = SimpleVectorStore()
store.add(embeddings, chunks)

print("Vector store successfully created.")


# =========================================================
# Step 5 – Initialize RAG System
# =========================================================

print_section("STEP 5 – Initializing RAG Engine")

rag = SimpleRAG(store)

print("RAG system is ready.")


# =========================================================
# Step 6 – Interactive Question Answering
# =========================================================

print_section("INTERACTIVE AI ASSISTANT")
print("Type 'exit' to stop.\n")

while True:

    query = input("\nAsk your academic question: ")

    if query.lower() == "exit":
        print("\nSession ended.")
        break

    answer = rag.answer(query)

    print("\n--- AI RESPONSE ---\n")
    pretty_print(answer)
    print("\n-------------------")
