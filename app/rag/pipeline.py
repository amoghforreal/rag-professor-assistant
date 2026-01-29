from app.embeddings.embedder import embed_texts
from app.vectorstore.store import SimpleVectorStore

class SimpleRAG:
    def __init__(self, vector_store: SimpleVectorStore):
        self.vector_store = vector_store

    def answer(self, query: str) -> str:
        """
        Retrieve relevant context and generate an answer.
        """
        query_embedding = embed_texts([query])[0]
        relevant_chunks = self.vector_store.similarity_search(query_embedding)

        if not relevant_chunks:
            return "No relevant information found in documents."

        context = "\n".join(relevant_chunks)

        # Placeholder generation logic
        response = f"Based on the documents, here is the relevant information:\n{context}"
        return response
