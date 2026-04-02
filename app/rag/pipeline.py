import numpy as np
from app.embeddings.embedder import embed_texts
from app.vectorstore.store import SimpleVectorStore


class SimpleRAG:
    def __init__(self, vector_store: SimpleVectorStore):
        self.vector_store = vector_store

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # -----------------------------
    # Detect user intent
    # -----------------------------
    def detect_intent(self, query: str):
        q = query.lower()

        if "bullet" in q or "points" in q or "list" in q:
            return "bullet"
        elif "explain" in q:
            return "explain"
        elif "summary" in q or "summarize" in q:
            return "summary"
        elif "define" in q or "what is" in q:
            return "definition"
        else:
            return "general"

    # -----------------------------
    # Clean keywords (IMPORTANT FIX)
    # -----------------------------
    def extract_keywords(self, query):
        stopwords = {
            "give", "tell", "me", "what", "is", "are", "in",
            "the", "a", "an", "of", "about", "please",
            "explain", "define", "summary", "summarize",
            "points", "bullet", "list"
        }

        words = query.lower().split()
        keywords = [w for w in words if w not in stopwords]

        return keywords

    # -----------------------------
    # Extract relevant sentences
    # -----------------------------
    def extract_relevant_sentences(self, chunks, query):
        keywords = self.extract_keywords(query)

        selected = []

        for chunk in chunks:
            sentences = chunk.split(".")
            for sentence in sentences:
                sentence_lower = sentence.lower()

                # If no keywords → accept everything
                if not keywords:
                    selected.append(sentence.strip())
                else:
                    if any(word in sentence_lower for word in keywords):
                        selected.append(sentence.strip())

        return list(dict.fromkeys(selected))

    # -----------------------------
    # Main Answer Function
    # -----------------------------
    def answer(self, query: str) -> str:

        query_embedding = embed_texts([query])[0]

        relevant_chunks = self.vector_store.similarity_search(query_embedding, top_k=3)

        if not relevant_chunks:
            return "No relevant academic information found."

        # Relaxed similarity check (IMPORTANT FIX)
        best_chunk_embedding = embed_texts([relevant_chunks[0]])[0]
        similarity_score = self.cosine_similarity(query_embedding, best_chunk_embedding)

        if similarity_score < 0.15:
            return "The question appears unrelated to the uploaded academic material."

        # Extract useful sentences
        filtered_sentences = self.extract_relevant_sentences(relevant_chunks, query)

        # Fallback if nothing matched
        if not filtered_sentences:
            filtered_sentences = relevant_chunks

        intent = self.detect_intent(query)

        # -----------------------------
        # Format Output
        # -----------------------------
        if intent == "bullet":
            response = "Answer in bullet points:\n\n"
            for sentence in filtered_sentences[:5]:
                if sentence:
                    response += f"• {sentence}\n"
            return response

        elif intent == "summary":
            return "Summary:\n\n" + " ".join(filtered_sentences[:3])

        elif intent == "definition":
            return "Definition:\n\n" + filtered_sentences[0]

        elif intent == "explain":
            return "Explanation:\n\n" + " ".join(filtered_sentences[:4])

        else:
            return "Answer:\n\n" + " ".join(filtered_sentences[:3])