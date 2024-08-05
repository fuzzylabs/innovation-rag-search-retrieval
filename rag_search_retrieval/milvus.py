"""Milvus module."""

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from milvus_model.base import BaseEmbeddingFunction
from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)
from scipy.sparse import csr_array
from sentence_transformers import SentenceTransformer


def sparse_to_dict(sparse_array: csr_array) -> dict[int, float]:
    """Convert sparse array in csr format to a dictionary.

    Args:
        sparse_array (csr_array): Sparse array in csr format

    Returns:
        dict[int, float]: Dictionary mapping tokens to score.
    """
    row_indices, col_indices = sparse_array.nonzero()
    non_zero_values = sparse_array.data
    result_dict = {}
    for col_index, value in zip(col_indices, non_zero_values):
        result_dict[col_index] = value
    return result_dict


def create_schema(dense_dim: int) -> CollectionSchema:
    """_summary_.

    Args:
        dense_dim (int): _description_

    Returns:
        CollectionSchema: _description_
    """
    # Specify the data schema for the new Collection.
    fields = [
        # Use auto generated id as primary key
        FieldSchema(
            name="pk",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=True,
            max_length=100,
        ),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="full_text_search_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
        FieldSchema(name="metadata", dtype=DataType.JSON),
    ]
    schema = CollectionSchema(fields, "")
    return schema


class CustomMilvusClient:
    """Custom Milvus client."""

    def __init__(self) -> None:
        """Connect to milvus lite client."""
        self.milvus_client = MilvusClient(uri="./milvus.db")

    def create_collection(
        self, collection_name: str, dense_dim: int, dense_distance_metric: str
    ) -> None:
        """Create a collection for a milvus vector store.

        Args:
            collection_name (str): Name of the collection
            dense_dim (int): Embedding dimension for dense vector
            dense_distance_metric (str): Metric used by dense embedding model
        """
        # Drop if collection exists
        has_collection = self.milvus_client.has_collection(collection_name, timeout=5)
        if has_collection:
            self.milvus_client.drop_collection(collection_name)

        index_params = self.milvus_client.prepare_index_params()
        # scalar index
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        # scalar index
        index_params.add_index(
            field_name="full_text_search_vector",
            index_name="sparse_inverted_index_full_text",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        # vector index
        index_params.add_index(
            field_name="dense_vector",
            index_name="flat",
            index_type="FLAT",
            metric_type=dense_distance_metric,
        )
        schema = create_schema(dense_dim)

        self.milvus_client.create_collection(
            collection_name,
            schema=schema,
            index_params=index_params,
        )

    def store(
        self,
        docs: Document,
        collection_name: str,
        dense_model: SentenceTransformer,
        sparse_model: BaseEmbeddingFunction,
        full_text_search_document_embeddings: csr_array,
        batch_size: int = 32,
    ) -> None:
        """Store texts, embedding (dense and sparse) and metadata.

        Args:
            docs (Document): Dataset to be stored in milvus vector store.
            collection_name (str): Name of the collection
            dense_model (SentenceTransformer): The embedding model used
            sparse_model (BaseEmbeddingFunction): The sparse model used
            full_text_search_document_embeddings (csr_array): The embeddings
                from BM25 model all the documents
            batch_size (int, optional): Batch size. Defaults to 32.
        """
        texts, metadatas = [], []
        for doc in docs:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        dense_embeddings = dense_model.encode(texts, batch_size=batch_size).tolist()
        sparse_arrays = sparse_model.encode_documents(documents=texts)
        sparse_embeddings = [
            sparse_to_dict(sparse_array) for sparse_array in sparse_arrays["sparse"]
        ]
        full_text_embeddings = [
            sparse_to_dict(sparse_array)
            for sparse_array in full_text_search_document_embeddings
        ]

        dataframe = pd.DataFrame(
            {
                "text": texts,
                "sparse_vector": sparse_embeddings,
                "dense_vector": dense_embeddings,
                "metadata": metadatas,
                "full_text_search_vector": full_text_embeddings,
            }
        )
        data = dataframe.to_dict("records")

        _ = self.milvus_client.insert(collection_name, data, progress_bar=True)

    def dense_search(
        self,
        collection_name: str,
        query_dense_embedding: np.ndarray,
        top_k: int,
        dense_search_params: dict,
    ) -> list:
        """Perform dense search.

        Args:
            collection_name (str): Name of the collection
            query_dense_embedding (np.ndarray): The embedding vector
                from the dense model for the query
            top_k (int): Top k entries semantically similar to query
                in the vector store.
            dense_search_params (dict): The search parameters such as
                metrics and other param used to calculate score.

        Returns:
            list: List of text, index and score sorted by score.
        """
        result = self.milvus_client.search(
            collection_name=collection_name,
            data=query_dense_embedding,
            anns_field="dense_vector",
            limit=top_k,
            output_fields=["text"],
            search_params=dense_search_params,
        )
        return result[0]

    def sparse_search(
        self,
        collection_name: str,
        query_sparse_embedding: list[dict[int, float]],
        top_k: int,
        sparse_search_params: dict,
    ) -> list:
        """_summary_.

        Args:
            collection_name (str): Name of the collection
            query_sparse_embedding (list[dict[int, float]]):
                The sparse vector from the sparse model for the query
            top_k (int): Top k entries semantically similar to query
                in the vector store.
            sparse_search_params (dict): The search parameters such as
                metrics and other param used to calculate score.

        Returns:
            list: List of text, index and score sorted by score.
        """
        result = self.milvus_client.search(
            collection_name=collection_name,
            data=[query_sparse_embedding],
            anns_field="sparse_vector",
            limit=top_k,
            output_fields=["text"],
            search_params=sparse_search_params,
        )
        return result[0]

    def full_text_search(
        self,
        collection_name: str,
        query_full_text_embedding: list[dict[int, float]],
        top_k: int,
        sparse_search_params: dict,
    ) -> list:
        """Perform full text search using BM25 model.

        Args:
            collection_name (str): Name of the collection
            query_full_text_embedding (list[dict[int, float]]):
                The full text vector from the BM25 model for the query
            top_k (int): Top k entries semantically similar to query
                in the vector store.
            sparse_search_params (dict): The search parameters such as
                metrics and other param used to calculate score.

        Returns:
            list: List of text, index and score sorted by score.
        """
        result = self.milvus_client.search(
            collection_name=collection_name,
            data=[query_full_text_embedding],
            anns_field="full_text_search_vector",
            limit=top_k,
            output_fields=["text"],
            search_params=sparse_search_params,
        )
        return result[0]
