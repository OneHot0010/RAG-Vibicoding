"""Vector store abstractions and factory helpers."""

from libs.vector_store.base_vector_store import BaseVectorStore, VectorQueryResult, VectorRecord
from libs.vector_store.chroma_store import ChromaStore, ChromaStoreError
from libs.vector_store.vector_store_factory import VectorStoreFactory, VectorStoreFactoryError

VectorStoreFactory.register("chroma", ChromaStore)

__all__ = [
    "BaseVectorStore",
    "ChromaStore",
    "ChromaStoreError",
    "VectorQueryResult",
    "VectorRecord",
    "VectorStoreFactory",
    "VectorStoreFactoryError",
]
