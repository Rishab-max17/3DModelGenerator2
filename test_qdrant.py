import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import numpy as np
import uuid

# Load environment variables
load_dotenv()

# Get Qdrant configuration from environment variables
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = os.getenv("QDRANT_COLLECTION", "3d_models")

print(f"Connecting to Qdrant at: {qdrant_url}")
print(f"Using collection: {collection_name}")

try:
    # Initialize the client
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key
    )
    
    # Test the connection by getting the list of collections
    collections = client.get_collections().collections
    print(f"Successfully connected to Qdrant!")
    print(f"Available collections: {[c.name for c in collections]}")
    
    # Create collection if it doesn't exist
    if collection_name not in [c.name for c in collections]:
        print(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )
        print(f"Collection '{collection_name}' created successfully!")
    else:
        print(f"Collection '{collection_name}' already exists.")
    
    # Try adding a test vector with UUID
    test_id = str(uuid.uuid4())
    test_vector = np.random.rand(512).astype(np.float32)
    test_metadata = {
        "description": "Test 3D model vector",
        "timestamp": "2023-04-10T12:00:00Z",
        "test": True
    }
    
    print(f"Adding test vector with UUID: {test_id}")
    
    client.upsert(
        collection_name=collection_name,
        points=[
            {
                "id": test_id,
                "vector": test_vector.tolist(),
                "payload": test_metadata
            }
        ]
    )
    
    print(f"Successfully added test vector with ID: {test_id}")
    
    # Retrieve the vector to confirm
    results = client.retrieve(
        collection_name=collection_name,
        ids=[test_id]
    )
    
    if results:
        print(f"Successfully retrieved test vector: {results[0].id}")
        print(f"Metadata: {results[0].payload}")
        
        # Perform a search to test similarity search
        search_results = client.search(
            collection_name=collection_name,
            query_vector=test_vector.tolist(),
            limit=5
        )
        
        print(f"Search results count: {len(search_results)}")
        for i, result in enumerate(search_results):
            print(f"  {i+1}. ID: {result.id}, Score: {result.score}")
    else:
        print("Failed to retrieve test vector")
    
except Exception as e:
    print(f"Error: {e}") 