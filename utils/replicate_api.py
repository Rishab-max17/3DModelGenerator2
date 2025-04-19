import os
import time
import replicate
import numpy as np
from io import BytesIO
from pathlib import Path
import base64
from typing import Dict, List, Optional, Union, Tuple
from PIL import Image
import requests
from dotenv import load_dotenv
import uuid
from loguru import logger
import tempfile
import json
import sys
import traceback
import random
import backoff

# Load environment variables
load_dotenv()

# Constants
MODEL_ID = "tencent/hunyuan3d-2:b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb"
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Only use the verified working model version
ALTERNATE_MODEL_VERSION = "b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb"

# Configure logger for better debugging
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("replicate_api.log", rotation="10 MB", retention="1 week")

def image_to_base64(image_path):
    """
    Convert an image to base64 string
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string
    """
    if isinstance(image_path, str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image_path, Image.Image):
        # If PIL Image is provided, save to temp file first
        temp_path = f"/tmp/{uuid.uuid4()}.png"
        image_path.save(temp_path)
        with open(temp_path, "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode('utf-8')
        os.remove(temp_path)  # Clean up
        return base64_str
    else:
        raise ValueError("Invalid image format. Must be path string or PIL Image.")

# Add an exponential backoff decorator for API calls
@backoff.on_exception(backoff.expo, 
                     (requests.exceptions.RequestException, 
                      Exception), 
                     max_tries=5,
                     jitter=backoff.full_jitter)
def download_file(url, output_path):
    """
    Download a file from URL to the specified path with retry logic
    
    Args:
        url: URL to download from
        output_path: Path where the file should be saved
        
    Returns:
        Path to the downloaded file
    """
    logger.info(f"Downloading file from {url} to {output_path}")
    response = requests.get(url, stream=True, timeout=120)  # Increased timeout
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                # Log progress for large files
                if total_size > 5*1024*1024 and downloaded % (5*1024*1024) < block_size:  # Log every 5MB
                    logger.debug(f"Download progress: {downloaded/total_size*100:.1f}%")
    
    logger.info(f"Successfully downloaded file to {output_path}")
    return output_path

def validate_api_key():
    """Validate that the Replicate API key is set and valid"""
    if not REPLICATE_API_TOKEN or REPLICATE_API_TOKEN == "your_new_token_here":
        error_msg = "REPLICATE_API_TOKEN is not set or invalid in .env file. Please get a valid token from https://replicate.com/account/api-tokens"
        logger.error(error_msg)
        return False
    
    # Test the API key with a simple call
    try:
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
        # Just get model info to verify the key works
        model_info = replicate.models.get("tencent/hunyuan3d-2")
        logger.info(f"API key validated successfully. Model: {model_info.name}")
        return True
    except Exception as e:
        logger.error(f"API key validation failed: {e}")
        return False

def generate_3d_model(
    image=None,
    image_path=None, 
    output_dir=None, 
    request_id=None,
    format="glb",
    prompt=None,
    resolution=256,
    use_high_quality=False,
    model_version="b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb",
    max_retries=3,
    texture_resolution=2048,
    remove_background=True,
    geometry_detail_level="high",
    texture_quality="high"
):
    """
    Generate a 3D model from an image using Replicate API with enhanced error handling.
    
    Args:
        image: Input image path or PIL Image (will be used if image_path is None)
        image_path: Path to the input image (deprecated, use image instead)
        output_dir: Directory to save the output files
        request_id: Unique ID for tracking this request
        format: Output format (glb or obj)
        prompt: Optional text prompt for guidance
        resolution: Resolution for the 3D model
        use_high_quality: Whether to use high quality settings
        model_version: Replicate API model version
        max_retries: Maximum number of retries for API calls
        texture_resolution: Resolution of textures (higher = better quality)
        remove_background: Whether to automatically remove background from image
        geometry_detail_level: Level of geometric detail in the model
        texture_quality: Quality level for textures
        
    Returns:
        Dict with model_path key pointing to the generated 3D model file
    """
    logger.info(f"Starting 3D model generation for request {request_id}")
    logger.info(f"Parameters: format={format}, resolution={resolution}, use_high_quality={use_high_quality}, model_version={model_version}")
    
    # Validate API key
    if not validate_api_key():
        error_msg = "Invalid or missing Replicate API token"
        logger.error(error_msg)
        # Create a fallback model with clear indication it's a fallback
        try:
            import trimesh
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            fallback_path = output_dir / f"model.{format}"
            
            # Create a more informative fallback model with text
            if format.lower() == "glb":
                # For GLB, create a simple box
                fallback_mesh = trimesh.creation.box(extents=[1, 0.5, 0.1])
                # Fix vertex colors
                num_vertices = len(fallback_mesh.vertices)
                vertex_colors = np.ones((num_vertices, 4), dtype=np.uint8) * [255, 200, 200, 255]
                fallback_mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
            else:
                # For OBJ, create a sphere
                fallback_mesh = trimesh.creation.icosphere()
                # Fix vertex colors
                num_vertices = len(fallback_mesh.vertices)
                vertex_colors = np.ones((num_vertices, 4), dtype=np.uint8) * [200, 200, 255, 255]
                fallback_mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
                
            fallback_mesh.export(fallback_path)
            logger.warning(f"Created fallback model due to API issues: {fallback_path}")
            
            # Return the fallback model path with an error flag
            return {
                "model_path": fallback_path,
                "error": error_msg,
                "is_fallback": True
            }
        except Exception as fallback_error:
            logger.error(f"Failed to create fallback model: {fallback_error}")
            raise ValueError(error_msg)
    
    # Set API token in environment
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
    
    # Handle the image parameter
    if image is not None:
        image_path = image
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save input image for debugging and reference
    input_img_path = output_dir / "input_image.png"
    if isinstance(image_path, Image.Image):
        logger.info(f"Input is PIL Image, saving to {input_img_path}")
        image_path.save(input_img_path)
        img_for_api = input_img_path
    elif not isinstance(image_path, str) and not isinstance(image_path, Path):
        logger.error(f"Invalid image type: {type(image_path)}")
        raise ValueError(f"Invalid image type: {type(image_path)}")
    else:
        try:
            logger.info(f"Input is path: {image_path}")
            img = Image.open(image_path)
            img.save(input_img_path)
            img_for_api = input_img_path
        except Exception as e:
            logger.error(f"Error saving input image: {e}")
            img_for_api = image_path
    
    # Create a fallback model as a placeholder - will be overwritten if API succeeds
    fallback_path = output_dir / f"model.{format}"
    try:
        import trimesh
        placeholder_mesh = trimesh.creation.icosphere(subdivisions=3)
        # Fix the vertex_colors assignment - it needs to match the dimensionality of vertices
        # Get the number of vertices in the mesh
        num_vertices = len(placeholder_mesh.vertices)
        # Create a vertex color array with the right shape
        vertex_colors = np.ones((num_vertices, 4), dtype=np.uint8) * [220, 220, 220, 255]
        placeholder_mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
        placeholder_mesh.export(fallback_path)
        logger.info(f"Created placeholder model at {fallback_path}")
    except Exception as fallback_error:
        logger.error(f"Could not create placeholder model: {fallback_error}")
    
    # Implement retry logic 
    for retry in range(max_retries):
        try:
            logger.info(f"API request attempt {retry+1}/{max_retries}")
            
            # Add some randomization to the model version if retry > 0
            current_model_version = model_version
            if retry > 0 and random.random() < 0.5:
                # Try alternate model version on some retries
                # Only use verified working model version
                alternate_versions = [
                    "b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb",
                    ALTERNATE_MODEL_VERSION  # Use the known working version
                ]
                alternate_version = [v for v in alternate_versions if v != model_version][0]
                current_model_version = alternate_version
                logger.info(f"Trying alternate model version: {current_model_version}")
            
            # Prepare input data with enhanced parameters
            input_data = {
                "image": open(str(img_for_api), "rb")
            }
            
            # Add optional parameters
            if prompt:
                # Enhance prompt for better 3D generation
                enhanced_prompt = prompt
                if not any(keyword in prompt.lower() for keyword in ["3d", "three dimensional", "detailed", "high quality"]):
                    enhanced_prompt = f"{prompt}, detailed 3D model with clear shape and texture, high quality"
                    logger.info(f"Enhanced prompt: '{prompt}' -> '{enhanced_prompt}'")
                else:
                    enhanced_prompt = prompt
                
                logger.info(f"Using enhanced prompt: {enhanced_prompt}")
                input_data["prompt"] = enhanced_prompt
                
            if use_high_quality:
                logger.info("Setting high quality flag")
                input_data["high_quality"] = "true"
                
            # Add format
            input_data["format"] = format
            
            # Add enhanced parameters
            if remove_background:
                logger.info("Setting remove_background flag")
                input_data["remove_background"] = "true"
                
            if texture_resolution > 1024:
                logger.info(f"Setting texture resolution to {texture_resolution}")
                input_data["texture_resolution"] = str(texture_resolution)
                
            if geometry_detail_level in ["low", "medium", "high"]:
                logger.info(f"Setting geometry detail level to {geometry_detail_level}")
                input_data["geometry_detail"] = geometry_detail_level
                
            if texture_quality in ["basic", "medium", "high"]:
                logger.info(f"Setting texture quality to {texture_quality}")
                input_data["texture_quality"] = texture_quality
            
            # Run the prediction with safe error handling
            prediction = None
            try:
                logger.info(f"Submitting prediction to replicate model: tencent/hunyuan3d-2:{current_model_version}")
                logger.info(f"Input parameters: {json.dumps({k: str(v) if k != 'image' else '<image_data>' for k, v in input_data.items()})}")
                
                # Enhanced API call with all parameters
                prediction = replicate.run(
                    f"tencent/hunyuan3d-2:{current_model_version}",
                    input={
                        "image": open(str(img_for_api), "rb"),
                        "prompt": enhanced_prompt if prompt else None,
                        "high_quality": "true" if use_high_quality else None,
                        "format": format,
                        "remove_background": "true" if remove_background else None,
                        "texture_resolution": str(texture_resolution) if texture_resolution > 1024 else None,
                        "geometry_detail": geometry_detail_level if geometry_detail_level in ["low", "medium", "high"] else None,
                        "texture_quality": texture_quality if texture_quality in ["basic", "medium", "high"] else None
                    }
                )
                logger.info(f"Prediction completed successfully: {prediction}")
            except Exception as prediction_error:
                logger.error(f"Error in replicate.run: {prediction_error}")
                logger.error(traceback.format_exc())
                # Re-raise to be caught by the retry logic
                raise
            
            # Extract the model URL
            model_url = None
            if isinstance(prediction, list) and len(prediction) > 0:
                model_url = prediction[0]
            elif isinstance(prediction, dict):
                model_url = prediction.get("mesh") or prediction.get("3d_model") or prediction.get("model") or prediction.get("output")
            else:
                model_url = prediction
                
            if not model_url:
                logger.error(f"Could not find model URL in prediction result: {prediction}")
                raise ValueError(f"Could not find model URL in prediction result: {prediction}")
            
            # Download the model file
            output_path = output_dir / f"model.{format}"
            try:
                logger.info(f"Downloading 3D model from {model_url} to {output_path}")
                
                # Determine correct file extension from URL
                url_extension = Path(model_url).suffix.lower()
                if url_extension and url_extension[1:] in ['glb', 'obj']:
                    # Use the extension from the URL if it's valid
                    logger.info(f"URL indicates format: {url_extension[1:]}, requested format: {format}")
                    actual_output_path = output_dir / f"model{url_extension}"
                else:
                    # Fallback to requested format
                    actual_output_path = output_path
                
                # Download the file
                download_file(model_url, actual_output_path)
                logger.info(f"Download completed successfully: {actual_output_path}")
                
                # If formats don't match, try to convert (future enhancement)
                
                # Verify file size as a basic check
                file_size = actual_output_path.stat().st_size
                if file_size < 1000:  # Less than 1KB is suspicious
                    logger.warning(f"Downloaded file is suspiciously small: {file_size} bytes")
                    if retry < max_retries - 1:
                        logger.info("Retrying due to small file size")
                        continue
                
                return {"model_path": actual_output_path}
            except Exception as download_error:
                logger.error(f"Error downloading model file: {download_error}")
                if retry < max_retries - 1:
                    logger.info("Retrying after download error")
                    continue
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"Error on attempt {retry+1}/{max_retries}: {e}")
            logger.error(traceback.format_exc())
            
            # Sleep with exponential backoff before retry
            if retry < max_retries - 1:
                sleep_time = 2 ** retry + random.uniform(0, 1)  # Exponential backoff with jitter
                logger.info(f"Sleeping for {sleep_time:.2f} seconds before retry")
                time.sleep(sleep_time)
    
    # If we get here, all retries failed but we have a fallback
    logger.error(f"All {max_retries} attempts failed. Using fallback model.")
    if fallback_path.exists():
        return {"model_path": fallback_path, "is_fallback": True}
    
    # Last resort - create a new fallback
    try:
        import trimesh
        fallback_mesh = trimesh.creation.box()
        # Fix vertex colors here as well
        num_vertices = len(fallback_mesh.vertices)
        vertex_colors = np.ones((num_vertices, 4), dtype=np.uint8) * [255, 200, 200, 255]
        fallback_mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
        fallback_mesh.export(fallback_path)
        logger.warning(f"Using emergency fallback model")
        return {"model_path": fallback_path, "is_fallback": True}
    except Exception as last_error:
        logger.critical(f"Critical failure: {last_error}")
        raise ValueError(f"Failed to generate 3D model after {max_retries} attempts and fallback creation also failed") 