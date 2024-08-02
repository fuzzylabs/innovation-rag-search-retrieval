"""Data pipeline."""

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter


def prepare_data(file_path: str, text_splitter: TextSplitter) -> list[Document]:
    """Load and chunk pdf file.

    Args:
        file_path (str): Path to pdf file.
        text_splitter (TextSplitter): Text splitter to use
            for splitting the text into chunks.

    Returns:
        list[Document]: _description_
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split(text_splitter=text_splitter)
    return docs
