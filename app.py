import os
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import uuid
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import json

from model.generator import ModelGenerator
from utils.image_processing import preprocess_image
from utils.file_utils import save_upload, cleanup_files
from utils.vector_db import get_qdrant_client, add_model_to_database, search_similar_models, get_model_by_id

# Load environment variables
load_dotenv()

# Create upload and output directories
Path("uploads").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

app = FastAPI(
    title="2D to 3D Model Generator",
    description="Convert 2D images to 3D models using deep learning",
    version="1.0.0"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model_generator = ModelGenerator()

# Initialize Qdrant client
qdrant_client = get_qdrant_client()

class GenerationParams(BaseModel):
    resolution: int = 256
    use_high_quality: bool = False
    format: str = "glb"  # Output format (obj, glb, etc.)
    multi_view: bool = True  # Generate from multiple viewpoints
    use_hunyuan: bool = True  # Whether to use Hunyuan3D-2 model

class TextPromptParams(BaseModel):
    prompt: str
    resolution: int = 256
    format: str = "glb"  # Output format (obj, glb, etc.)
    use_hunyuan: bool = True  # Whether to use Hunyuan3D-2 model

class SimilarModelsParams(BaseModel):
    query: str
    limit: int = 5

@app.post("/generate")
async def generate_3d_model(
    background_tasks: BackgroundTasks,
    params: GenerationParams = None,
    file: UploadFile = File(...)
):
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided")
    
    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Save uploaded image
    file_path = await save_upload(file, request_id)
    
    # Preprocess image
    processed_image = preprocess_image(file_path)
    
    # Process with model and generate 3D model
    if params and params.use_hunyuan:
        try:
            output_path = model_generator.generate_hunyuan3d(
                processed_image, 
                request_id, 
                resolution=params.resolution if params else 256,
                use_high_quality=params.use_high_quality if params else False,
                format=params.format if params else "glb"
            )
        except Exception as e:
            print(f"Error using Hunyuan3D-2: {e}")
            print("Falling back to original implementation...")
            output_path = model_generator.generate(
                processed_image, 
                request_id, 
                resolution=params.resolution if params else 256,
                use_high_quality=params.use_high_quality if params else False,
                format=params.format if params else "glb",
                multi_view=params.multi_view if params else True
            )
    else:
        output_path = model_generator.generate(
            processed_image, 
            request_id, 
            resolution=params.resolution if params else 256,
            use_high_quality=params.use_high_quality if params else False,
            format=params.format if params else "glb",
            multi_view=params.multi_view if params else True
        )
    
    # Store model metadata in Qdrant
    try:
        # Create metadata
        metadata = {
            "filename": file.filename,
            "original_image": f"/models/{request_id}/input_image.png",
            "model_path": str(output_path),
            "resolution": params.resolution if params else 256,
            "use_high_quality": params.use_high_quality if params else False,
            "format": params.format if params else "glb",
            "multi_view": params.multi_view if params else True,
            "use_hunyuan": params.use_hunyuan if params else True,
            "creation_time": str(Path(output_path).stat().st_ctime if output_path.exists() else 0)
        }
        
        # Get or generate embedding
        try:
            from utils.model_embeddings import embedding_generator
            embedding = embedding_generator.generate_embedding_from_image(file_path)
        except (ImportError, Exception) as e:
            print(f"Error generating embedding: {e}")
            # Fallback to random embedding
            embedding = np.random.random(512)
            embedding = embedding / np.linalg.norm(embedding)
        
        # Store in vector database
        add_model_to_database(qdrant_client, request_id, embedding, metadata)
    except Exception as e:
        print(f"Error storing model in vector database: {e}")
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_files, request_id)
    
    return {
        "request_id": request_id,
        "model_url": f"/models/{request_id}/download",
        "status": "success"
    }

@app.post("/generate-with-prompt")
async def generate_with_prompt(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = Form(...),
    resolution: int = Form(256),
    format: str = Form("glb"),
    use_hunyuan: bool = Form(True)
):
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided")
    
    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Save uploaded image
    file_path = await save_upload(file, request_id)
    
    # Preprocess image
    processed_image = preprocess_image(file_path)
    
    # Process with model and generate 3D model with text prompt
    if use_hunyuan:
        try:
            output_path = model_generator.generate_with_text_prompt_hunyuan3d(
                processed_image,
                prompt,
                request_id,
                resolution=resolution,
                format=format
            )
        except Exception as e:
            print(f"Error using Hunyuan3D-2 with text prompt: {e}")
            print("Falling back to original implementation...")
            output_path = model_generator.generate_with_text_prompt(
                processed_image,
                prompt,
                request_id,
                resolution=resolution,
                format=format,
                multi_view=True
            )
    else:
        output_path = model_generator.generate_with_text_prompt(
            processed_image,
            prompt,
            request_id,
            resolution=resolution,
            format=format,
            multi_view=True
        )
    
    # Store model metadata in Qdrant
    try:
        # Create metadata
        metadata = {
            "filename": file.filename,
            "original_image": f"/models/{request_id}/input_image.png",
            "model_path": str(output_path),
            "prompt": prompt,
            "resolution": resolution,
            "format": format,
            "use_hunyuan": use_hunyuan,
            "creation_time": str(Path(output_path).stat().st_ctime if output_path.exists() else 0)
        }
        
        # Get or generate embedding
        try:
            from utils.model_embeddings import embedding_generator
            # For text prompt models, combine image and text embeddings
            image_embedding = embedding_generator.generate_embedding_from_image(file_path)
            text_embedding = embedding_generator.generate_embedding_from_metadata({"prompt": prompt})
            embedding = (image_embedding + text_embedding) / 2
            embedding = embedding / np.linalg.norm(embedding)
        except (ImportError, Exception) as e:
            print(f"Error generating embedding: {e}")
            # Fallback to random embedding
            embedding = np.random.random(512)
            embedding = embedding / np.linalg.norm(embedding)
        
        # Store in vector database
        add_model_to_database(qdrant_client, request_id, embedding, metadata)
    except Exception as e:
        print(f"Error storing model in vector database: {e}")
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_files, request_id)
    
    return {
        "request_id": request_id,
        "model_url": f"/models/{request_id}/download",
        "status": "success"
    }

@app.post("/generate-simple")
async def generate_simple(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    resolution: int = 256,
    use_high_quality: bool = False,
    format: str = "glb",
    multi_view: bool = True,
    use_hunyuan: bool = True
):
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided")
    
    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Save uploaded image
    file_path = await save_upload(file, request_id)
    
    # Preprocess image
    processed_image = preprocess_image(file_path)
    
    # Process with model and generate 3D model
    if use_hunyuan:
        try:
            output_path = model_generator.generate_hunyuan3d(
                processed_image, 
                request_id, 
                resolution=resolution,
                use_high_quality=use_high_quality,
                format=format
            )
        except Exception as e:
            print(f"Error using Hunyuan3D-2: {e}")
            print("Falling back to original implementation...")
            output_path = model_generator.generate(
                processed_image, 
                request_id, 
                resolution=resolution,
                use_high_quality=use_high_quality,
                format=format,
                multi_view=multi_view
            )
    else:
        output_path = model_generator.generate(
            processed_image, 
            request_id, 
            resolution=resolution,
            use_high_quality=use_high_quality,
            format=format,
            multi_view=multi_view
        )
    
    # Store model metadata in Qdrant
    try:
        # Create metadata
        metadata = {
            "filename": file.filename,
            "original_image": f"/models/{request_id}/input_image.png",
            "model_path": str(output_path),
            "resolution": resolution,
            "use_high_quality": use_high_quality,
            "format": format,
            "multi_view": multi_view,
            "use_hunyuan": use_hunyuan,
            "creation_time": str(Path(output_path).stat().st_ctime if output_path.exists() else 0)
        }
        
        # Get or generate embedding
        try:
            from utils.model_embeddings import embedding_generator
            embedding = embedding_generator.generate_embedding_from_image(file_path)
        except (ImportError, Exception) as e:
            print(f"Error generating embedding: {e}")
            # Fallback to random embedding
            embedding = np.random.random(512)
            embedding = embedding / np.linalg.norm(embedding)
        
        # Store in vector database
        add_model_to_database(qdrant_client, request_id, embedding, metadata)
    except Exception as e:
        print(f"Error storing model in vector database: {e}")
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_files, request_id)
    
    return {
        "request_id": request_id,
        "model_url": f"/models/{request_id}/download",
        "status": "success"
    }

@app.get("/models/{request_id}/download")
async def download_model(request_id: str):
    output_dir = Path("outputs") / request_id
    model_files = list(output_dir.glob("*.glb")) + list(output_dir.glob("*.obj"))
    
    if not model_files:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return FileResponse(
        model_files[0],
        media_type="application/octet-stream",
        filename=model_files[0].name
    )

@app.get("/models/{request_id}/depth_map.png")
async def get_depth_map(request_id: str):
    """Serve the depth map image for a model"""
    output_dir = Path("outputs") / request_id
    depth_map_path = output_dir / "depth_map.png"
    
    if not depth_map_path.exists():
        raise HTTPException(status_code=404, detail="Depth map not found")
    
    return FileResponse(
        depth_map_path,
        media_type="image/png",
        filename="depth_map.png"
    )

@app.get("/models/{request_id}/input_image.png")
async def get_input_image(request_id: str):
    """Serve the input image for a model"""
    output_dir = Path("outputs") / request_id
    input_image_path = output_dir / "input_image.png"
    
    if not input_image_path.exists():
        raise HTTPException(status_code=404, detail="Input image not found")
    
    return FileResponse(
        input_image_path,
        media_type="image/png",
        filename="input_image.png"
    )

@app.get("/")
async def read_root():
    return {"message": "2D to 3D Model Generator API"}

@app.get("/models/search")
async def search_models(
    query: str = Query(..., description="Search query text"),
    limit: int = Query(5, description="Maximum number of results to return")
):
    """
    Search for similar 3D models using text query
    """
    try:
        # Generate embedding from the query text
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = model.encode(query, normalize_embeddings=True)
        except (ImportError, Exception) as e:
            print(f"Error generating query embedding: {e}")
            # Fallback to random embedding
            query_embedding = np.random.random(512)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search for similar models
        results = search_similar_models(qdrant_client, query_embedding, limit=limit)
        
        # Format the results
        formatted_results = []
        for result in results:
            model_id = result["model_id"]
            score = result["score"]
            metadata = result["metadata"]
            
            formatted_results.append({
                "model_id": model_id,
                "similarity_score": score,
                "metadata": metadata,
                "download_url": f"/models/{model_id}/download"
            })
        
        return {
            "query": query,
            "results": formatted_results
        }
    except Exception as e:
        print(f"Error searching models: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching models: {str(e)}")

@app.get("/models/{request_id}/info")
async def get_model_info(request_id: str):
    """
    Get information about a specific model
    """
    try:
        # Retrieve model from vector database
        model_data = get_model_by_id(qdrant_client, request_id)
        
        if not model_data:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Return model info
        return {
            "model_id": model_data["model_id"],
            "metadata": model_data["metadata"],
            "download_url": f"/models/{request_id}/download"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True) 