def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Placeholder embedding function.
    Will be replaced by a real model later.
    """
    embeddings = []

    for text in texts:
        vector = [float(len(text))]  # dummy embedding
        embeddings.append(vector)

    return embeddings
