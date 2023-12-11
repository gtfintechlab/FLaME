# pip install nltk

import nltk

nltk.download("punkt")

from nltk.tokenize import word_tokenize


def split_document(document, max_tokens_per_chunk=1000):
    tokens = word_tokenize(document)

    chunks = []
    current_chunk = []

    for token in tokens:
        if len(current_chunk) + len(token) <= max_tokens_per_chunk:
            current_chunk.append(token)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [token]

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks