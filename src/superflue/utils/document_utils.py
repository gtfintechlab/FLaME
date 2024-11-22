# TODO: (Glenn) Evaluate code will need to be moved into its own folder not utils.

# TODO: (Glenn) I prefer to avoid using NLTK unless there's something we cannot get from another package
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")


def split_document(document, max_tokens_per_chunk=1000):
    """Splits a document into chunks based on the maximum number of tokens per chunk.

    Args:
        document (str): The document to be split.
        max_tokens_per_chunk (int): The maximum number of tokens allowed in each chunk.

    Returns:
        List[str]: A list of document chunks.
    """
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
