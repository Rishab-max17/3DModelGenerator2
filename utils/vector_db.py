import os
from typing import List, Dict, Optional, Union
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import PointStruct, Distance, VectorParams, OptimizersConfigDiff

# Load environment variables
load_dotenv()

# Get Qdrant configuration from environment variables or use defaults
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "3d_models")
VECTOR_SIZE = 512  # Size of the vector embeddings

def get_qdrant_client() -> QdrantClient:
    """
    Create and return a connection to the Qdrant vector database.
    
    Returns:
        QdrantClient: Connected client instance
    """
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        return QdrantClient(url=QDRANT_URL)

def ensure_collection_exists(client: QdrantClient, collection_name: str = QDRANT_COLLECTION_NAME) -> bool:
    """
    Check if the collection exists, and create it if it doesn't.
    
    Args:
        client: Qdrant client
        collection_name: Name of the collection
        
    Returns:
        bool: True if the collection exists or was created successfully
    """
    try:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            # Create new collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=10000  # Optional: Configure when to build index
                )
            )
            print(f"Created new collection: {collection_name}")
        return True
    except Exception as e:
        print(f"Error ensuring collection exists: {e}")
        return False

def add_model_to_database(
    client: QdrantClient,
    model_id: str,
    embedding: np.ndarray,
    metadata: Dict,
    collection_name: str = QDRANT_COLLECTION_NAME
) -> bool:
    """
    Add a 3D model embedding to the vector database.
    
    Args:
        client: Qdrant client
        model_id: Unique identifier for the model (request_id)
        embedding: Vector representation of the model
        metadata: Additional metadata about the model
        collection_name: Name of the collection
        
    Returns:
        bool: True if successful
    """
    try:
        # Ensure the collection exists
        ensure_collection_exists(client, collection_name)
        
        # Add the point to the collection
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=model_id,
                    vector=embedding.tolist(),
                    payload=metadata
                )
            ]
        )
        return True
    except Exception as e:
        print(f"Error adding model to database: {e}")
        return False

def search_similar_models(
    client: QdrantClient,
    query_vector: np.ndarray,
    limit: int = 5,
    collection_name: str = QDRANT_COLLECTION_NAME
) -> List[Dict]:
    """
    Search for similar 3D models in the database.
    
    Args:
        client: Qdrant client
        query_vector: Vector representation to search for
        limit: Maximum number of results to return
        collection_name: Name of the collection
        
    Returns:
        List[Dict]: List of similar models with their metadata
    """
    try:
        # Ensure the collection exists
        if not ensure_collection_exists(client, collection_name):
            return []
        
        # Search for similar vectors
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=limit
        )
        
        # Format the results
        results = []
        for scored_point in search_result:
            results.append({
                "model_id": scored_point.id,
                "score": scored_point.score,
                "metadata": scored_point.payload
            })
        
        return results
    except Exception as e:
        print(f"Error searching similar models: {e}")
        return []

def delete_model(
    client: QdrantClient,
    model_id: str,
    collection_name: str = QDRANT_COLLECTION_NAME
) -> bool:
    """
    Delete a model from the vector database.
    
    Args:
        client: Qdrant client
        model_id: ID of the model to delete
        collection_name: Name of the collection
        
    Returns:
        bool: True if successful
    """
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=rest.PointIdsList(
                points=[model_id]
            )
        )
        return True
    except Exception as e:
        print(f"Error deleting model: {e}")
        return False

def get_model_by_id(
    client: QdrantClient,
    model_id: str,
    collection_name: str = QDRANT_COLLECTION_NAME
) -> Optional[Dict]:
    """
    Retrieve a specific model by its ID.
    
    Args:
        client: Qdrant client
        model_id: ID of the model to retrieve
        collection_name: Name of the collection
        
    Returns:
        Optional[Dict]: Model data if found, None otherwise
    """
    try:
        result = client.retrieve(
            collection_name=collection_name,
            ids=[model_id]
        )
        
        if result and len(result) > 0:
            return {
                "model_id": result[0].id,
                "metadata": result[0].payload
            }
        return None
    except Exception as e:
        print(f"Error retrieving model: {e}")
        return None 