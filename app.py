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
import time

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
    texture_resolution: int = 2048  # Resolution of textures
    remove_background: bool = True  # Whether to remove background
    geometry_detail_level: str = "high"  # Level of geometric detail
    texture_quality: str = "high"  # Quality of textures

class TextPromptParams(BaseModel):
    prompt: str
    resolution: int = 256
    format: str = "glb"  # Output format (obj, glb, etc.)
    use_hunyuan: bool = True  # Whether to use Hunyuan3D-2 model
    texture_resolution: int = 2048  # Resolution of textures
    remove_background: bool = True  # Whether to remove background
    geometry_detail_level: str = "high"  # Level of geometric detail
    texture_quality: str = "high"  # Quality of textures

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
    
    # Override parameters for better quality models
    resolution = params.resolution if params else 256
    resolution = max(resolution, 512)  # Ensure minimum 512 resolution for good mesh structure
    use_high_quality = True  # Always use high quality regardless of user setting
    format = params.format if params else "glb"
    multi_view = True  # Always use multi-view for better 3D structure
    
    # Process with model and generate 3D model
    if params and params.use_hunyuan:
        try:
            print(f"Generating high-quality Hunyuan3D model for request {request_id}")
            output_path = model_generator.generate_hunyuan3d(
                processed_image, 
                request_id, 
                resolution=resolution,
                use_high_quality=use_high_quality,
                format=format,
                texture_resolution=params.texture_resolution,
                remove_background=params.remove_background,
                geometry_detail_level=params.geometry_detail_level,
                texture_quality=params.texture_quality
            )
        except Exception as e:
            print(f"Error using Hunyuan3D-2: {e}")
            print("Falling back to enhanced local implementation...")
            output_path = model_generator.generate(
                processed_image, 
                request_id, 
                resolution=resolution,
                use_high_quality=use_high_quality,
                format=format,
                multi_view=multi_view
            )
    else:
        print(f"Generating enhanced local 3D model for request {request_id}")
        output_path = model_generator.generate(
            processed_image, 
            request_id, 
            resolution=resolution,
            use_high_quality=use_high_quality,
            format=format,
            multi_view=multi_view
        )
    
    # Verify the model exists and has proper structure
    if not output_path.exists():
        raise HTTPException(status_code=500, detail="Failed to generate 3D model file")
    
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
    
    # Return enhanced response with mesh info
    return {
        "request_id": request_id,
        "model_url": f"/models/{request_id}/download",
        "status": "success",
        "mesh_info": {
            "resolution": resolution,
            "format": format,
            "file_size": output_path.stat().st_size if output_path.exists() else 0
        }
    }

@app.post("/generate-with-prompt")
async def generate_with_prompt(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = Form(...),
    resolution: int = Form(256),
    format: str = Form("glb"),
    use_hunyuan: bool = Form(True),
    retry_count: int = Form(2),
    model_version: str = Form("b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb"),
    texture_resolution: int = Form(2048),
    remove_background: bool = Form(True),
    geometry_detail_level: str = Form("high"),
    texture_quality: str = Form("high")
):
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided")
    
    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Save uploaded image
    file_path = await save_upload(file, request_id)
    
    # Preprocess image
    processed_image = preprocess_image(file_path)
    
    # Enhance prompt for better 3D generation if needed
    enhanced_prompt = prompt
    if not any(keyword in prompt.lower() for keyword in ["3d", "three dimensional", "detailed", "high quality"]):
        enhanced_prompt = f"{prompt}, detailed 3D model with clear shape and texture, high quality"
        print(f"Enhanced prompt: '{prompt}' -> '{enhanced_prompt}'")
    
    # Override parameters for better quality models
    resolution = max(resolution, 512)  # Ensure minimum 512 resolution for good mesh structure
    
    # Process with model and generate 3D model with text prompt
    if use_hunyuan:
        try:
            print(f"Generating high-quality Hunyuan3D model with text prompt for request {request_id}")
            output_path = model_generator.generate_with_text_prompt_hunyuan3d(
                processed_image,
                enhanced_prompt,
                request_id,
                resolution=resolution,
                format=format,
                model_version=model_version,
                retry_count=retry_count,
                texture_resolution=texture_resolution,
                remove_background=remove_background,
                geometry_detail_level=geometry_detail_level,
                texture_quality=texture_quality
            )
        except Exception as e:
            print(f"Error using Hunyuan3D-2 with text prompt: {e}")
            print("Falling back to enhanced local implementation...")
            output_path = model_generator.generate_with_text_prompt(
                processed_image,
                enhanced_prompt,
                request_id,
                resolution=resolution,
                format=format,
                multi_view=True
            )
    else:
        print(f"Generating enhanced local 3D model with text prompt for request {request_id}")
        output_path = model_generator.generate_with_text_prompt(
            processed_image,
            enhanced_prompt,
            request_id,
            resolution=resolution,
            format=format,
            multi_view=True
        )
    
    # Verify the model exists and has proper structure
    if not output_path.exists():
        raise HTTPException(status_code=500, detail="Failed to generate 3D model file")
    
    # Store model metadata in Qdrant
    try:
        # Create metadata
        metadata = {
            "filename": file.filename,
            "original_image": f"/models/{request_id}/input_image.png",
            "model_path": str(output_path),
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt if enhanced_prompt != prompt else None,
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
            text_embedding = embedding_generator.generate_embedding_from_metadata({"prompt": enhanced_prompt})
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
    
    # Return enhanced response with mesh info
    return {
        "request_id": request_id,
        "model_url": f"/models/{request_id}/download",
        "status": "success",
        "mesh_info": {
            "resolution": resolution,
            "format": format,
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt if enhanced_prompt != prompt else None,
            "file_size": output_path.stat().st_size if output_path.exists() else 0
        }
    }

@app.post("/generate-simple")
async def generate_simple(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    resolution: int = 256,
    use_high_quality: bool = False,
    format: str = "glb",
    multi_view: bool = True,
    use_hunyuan: bool = True,
    retry_count: int = 2,
    model_version: str = "b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb",
    texture_resolution: int = 2048,
    remove_background: bool = True,
    geometry_detail_level: str = "high",
    texture_quality: str = "high"
):
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided")
    
    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    
    try:
        # Save uploaded image
        file_path = await save_upload(file, request_id)
        
        # Preprocess image
        processed_image = preprocess_image(file_path)
        
        # Use enhanced quality settings for better mesh structure
        use_high_quality = True  # Always use high quality
        resolution = max(resolution, 512)  # Use at least 512 resolution for better detail
        
        # Process with model and generate 3D model
        if use_hunyuan:
            try:
                # Always use the most advanced model generation path
                print(f"Generating high-quality 3D model with Hunyuan3D for request {request_id}")
                output_path = model_generator.generate_hunyuan3d(
                    processed_image, 
                    request_id, 
                    resolution=resolution,
                    use_high_quality=use_high_quality,
                    format=format,
                    retry_count=retry_count,
                    model_version=model_version,
                    texture_resolution=texture_resolution,
                    remove_background=remove_background,
                    geometry_detail_level=geometry_detail_level,
                    texture_quality=texture_quality
                )
            except Exception as e:
                print(f"Error using Hunyuan3D-2: {e}")
                print("Falling back to original implementation with enhanced settings...")
                # Fall back to local generation with enhanced settings
                output_path = model_generator.generate(
                    processed_image, 
                    request_id, 
                    resolution=resolution,
                    use_high_quality=True,
                    format=format,
                    multi_view=True  # Always use multi-view for better structure
                )
        else:
            # For non-Hunyuan models, use enhanced settings
            print(f"Generating 3D model with local implementation for request {request_id}")
            output_path = model_generator.generate(
                processed_image, 
                request_id, 
                resolution=resolution,
                use_high_quality=True,  # Always use high quality
                format=format,
                multi_view=True  # Always use multi-view for better structure
            )
        
        # Verify model is valid and has proper mesh structure
        if not output_path.exists():
            raise HTTPException(status_code=500, detail="Failed to generate 3D model")
        
        # Store model metadata in Qdrant - wrapped with try/except and ignore errors
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
            try:
                add_model_to_database(qdrant_client, request_id, embedding, metadata)
            except Exception as db_error:
                print(f"Error adding to vector database, continuing without it: {db_error}")
        except Exception as e:
            print(f"Error in vector database operations, continuing anyway: {e}")
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_files, request_id)
        
        return {
            "request_id": request_id,
            "model_url": f"/models/{request_id}/download",
            "status": "success",
            "mesh_info": {
                "resolution": resolution,
                "format": format,
                "path": str(output_path),
                "file_size": output_path.stat().st_size
            }
        }
    except Exception as e:
        # Print full error for debugging
        import traceback
        print(f"Error in generate_simple: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating 3D model: {str(e)}")

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

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint that doesn't require external APIs"""
    return {
        "status": "success",
        "message": "API is working correctly",
        "timestamp": str(time.time())
    }

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
    Get information about a specific model, including 3D metrics
    """
    try:
        # Retrieve model from vector database
        model_data = get_model_by_id(qdrant_client, request_id)
        
        if not model_data:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Extract model path from metadata
        model_path = model_data["metadata"].get("model_path")
        
        # Get 3D metrics if the model file exists
        model_metrics = {}
        if model_path and Path(model_path).exists():
            try:
                import trimesh
                mesh = trimesh.load(model_path)
                
                # Extract mesh metrics
                model_metrics = {
                    "vertices_count": mesh.vertices.shape[0],
                    "faces_count": mesh.faces.shape[0],
                    "bounding_box": mesh.bounding_box.extents.tolist(),
                    "volume": float(mesh.volume) if hasattr(mesh, "volume") else None,
                    "surface_area": float(mesh.area) if hasattr(mesh, "area") else None,
                    "is_watertight": bool(mesh.is_watertight) if hasattr(mesh, "is_watertight") else None
                }
            except Exception as e:
                print(f"Error extracting mesh metrics: {e}")
                # Provide at least minimal info even if metrics extraction fails
                model_metrics = {
                    "vertices_count": -1,
                    "faces_count": -1
                }
        
        # Return model info with 3D metrics
        return {
            "model_id": model_data["model_id"],
            "metadata": model_data["metadata"],
            "download_url": f"/models/{request_id}/download",
            **model_metrics  # Include all mesh metrics
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003) 