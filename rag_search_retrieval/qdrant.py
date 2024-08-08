"""Qdrant vector store."""

import numpy as np
from fastembed.sparse.sparse_embedding_base import SparseTextEmbeddingBase
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from qdrant_client.conversions import common_types as types
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


class CustomQdrantClient:
    """Custom qdrant client."""

    def __init__(self) -> None:
        """Connect persist qdrant vector store to local path."""
        self.qdrant_client = QdrantClient(path="./qdrant.db")

    def create_collection(
        self,
        collection_name: str,
        dense_dim: int,
        dense_distance_metric: models.Distance,
    ) -> None:
        """Create a collection for a qdrant vector store.

        Args:
            collection_name (str): Name of the collection
            dense_dim (int): Embedding dimension for dense vector
            dense_distance_metric (models.Distance): Metric used
                by dense embedding model
        """
        # Drop if collection exists
        has_collection = self.qdrant_client.collection_exists(collection_name)
        if has_collection:
            self.qdrant_client.delete_collection(collection_name)

        # Create a collection with 1 dense vector and 2 sparse vector indexes
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_dim, distance=dense_distance_metric, on_disk=False
                )
            },
            sparse_vectors_config={
                "splade_sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    )
                ),
                "bm25_sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    )
                ),
            },
        )

    def store(
        self,
        docs: Document,
        collection_name: str,
        dense_model: SentenceTransformer,
        sparse_model: SparseTextEmbeddingBase,
        bm25_model: SparseTextEmbeddingBase,
        batch_size: int = 32,
    ) -> None:
        """Store texts, embedding (dense and sparse) and metadata.

        Args:
            docs (Document): Dataset to be stored in milvus vector store.
            collection_name (str): Name of the collection
            dense_model (SentenceTransformer): The embedding model used
            sparse_model (SparseTextEmbeddingBase): The sparse model used
            bm25_model (SparseTextEmbeddingBase): Bm25 model to use
                for full text search
            batch_size (int, optional): Batch size. Defaults to 32.
        """
        texts, metadatas = [], []
        for doc in docs:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        dense_embeddings = dense_model.encode(texts, batch_size=batch_size).tolist()
        sparse_embeddings = sparse_model.embed(documents=texts, batch_size=batch_size)
        full_text_embeddings = bm25_model.embed(documents=texts, batch_size=batch_size)
        sparse_embeddings = [embed for embed in sparse_embeddings]  # noqa: C416
        full_text_embeddings = [embed for embed in full_text_embeddings]  # noqa: C416

        points = [
            models.PointStruct(
                id=i + 1,
                vector={
                    "dense": dense_embeddings[i],
                    "splade_sparse": models.SparseVector(
                        indices=sparse_embeddings[i].indices,
                        values=sparse_embeddings[i].values,
                    ),
                    "bm25_sparse": models.SparseVector(
                        indices=full_text_embeddings[i].indices,
                        values=full_text_embeddings[i].values,
                    ),
                },
                payload={
                    "text": texts[i],
                    "metadata": metadatas[i],
                },
            )
            for i in range(len(texts))
        ]

        self.qdrant_client.upsert(collection_name=collection_name, points=points)

    def dense_search(
        self,
        collection_name: str,
        query_dense_embedding: np.ndarray,
        top_k: int,
    ) -> list:
        """Perform dense search.

        Args:
            collection_name (str): Name of the collection
            query_dense_embedding (np.ndarray): The embedding vector
                from the dense model for the query
            top_k (int): Top k entries semantically similar to query
                in the vector store.

        Returns:
            list: List of text, index and score sorted by score.
        """
        result = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=models.NamedVector(name="dense", vector=query_dense_embedding),
            limit=top_k,
        )
        return result

    def sparse_search(
        self,
        collection_name: str,
        query_sparse_embedding: types.SparseVector,
        top_k: int,
    ) -> list:
        """Perform sparse search.

        Args:
            collection_name (str): Name of the collection
            query_sparse_embedding (types.SparseVector): The embedding vector
                from the sparse embedding model for the query
            top_k (int): Top k entries similar to query in the vector store.

        Returns:
            list: List of text, index and score sorted by score.
        """
        result = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=models.NamedSparseVector(
                name="splade_sparse",
                vector=models.SparseVector(
                    indices=query_sparse_embedding.indices,
                    values=query_sparse_embedding.values,
                ),
            ),
            limit=top_k,
        )
        return result

    def full_text_search(
        self,
        collection_name: str,
        query_full_text_embedding: types.SparseVector,
        top_k: int,
    ) -> list:
        """Perform full text search using Bm25 model.

        Args:
            collection_name (str): Name of the collection
            query_full_text_embedding (types.SparseVector): The embedding vector
                from the sparse embedding model for the query
            top_k (int): Top k entries similar to query in the vector store.

        Returns:
            list: List of text, index and score sorted by score.
        """
        result = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=models.NamedSparseVector(
                name="bm25_sparse",
                vector=models.SparseVector(
                    indices=query_full_text_embedding.indices,
                    values=query_full_text_embedding.values,
                ),
            ),
            limit=top_k,
        )
        return result
