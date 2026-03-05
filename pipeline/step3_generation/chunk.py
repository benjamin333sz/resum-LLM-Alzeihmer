def split_chunks(ids, chunk_size):

    chunks = [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]

    if len(chunks) > 1 and len(chunks[-1]) < chunk_size // 2:
        chunks[-2].extend(chunks[-1])
        chunks.pop()

    return chunks
