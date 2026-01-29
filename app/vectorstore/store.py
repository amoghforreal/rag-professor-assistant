class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []

    def add(self, embeddings: list[list[float]], texts: list[str]):
        for emb, txt in zip(embeddings, texts):
            self.vectors.append(emb)
            self.texts.append(txt)

    def similarity_search(self, query_embedding: list[float], top_k: int = 3) -> list[str]:
        scores = []

        for emb, txt in zip(self.vectors, self.texts):
            score = abs(emb[0] - query_embedding[0])  # simple distance
            scores.append((score, txt))

        scores.sort(key=lambda x: x[0])
        return [txt for _, txt in scores[:top_k]]
