#pip install langchain

import langchain
import nltk
nltk.download('punkt')
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.document_loaders import TextLoader
'''def document_splitter(doc):
    length_tokens = len(doc.split())
    if (length_tokens > 5000):
        
    loader = TextLoader(doc)
    text = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 2000, chunk_overlap=50)
    return text_splitter.split_documents(text)'''

from nltk.tokenize import word_tokenize


def split_document(document, max_tokens_per_chunk=1000):
    tokens = word_tokenize(document)
    
    chunks = []
    current_chunk = []

    for token in tokens:
        if len(current_chunk) + len(token) <= max_tokens_per_chunk:
            current_chunk.append(token)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [token]

    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks