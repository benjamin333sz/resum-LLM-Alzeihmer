def split_chunks(ids: list[str], max_articles_per_generation: int | None = None):
    """
    Split a list of ids into chunks.

    If chunk_size is None:
        return a single chunk containing all ids.
    """

    if max_articles_per_generation is None:
        return [ids]

    chunks = [
        ids[i : i + max_articles_per_generation]
        for i in range(0, len(ids), max_articles_per_generation)
    ]

    if len(chunks) > 1 and len(chunks[-1]) < max_articles_per_generation // 2:
        chunks[-2].extend(chunks[-1])
        chunks.pop()

    return chunks
