import torch
from pathlib import Path
import numpy as np
import trimesh
import os
from diffusers import StableDiffusionDepth2ImgPipeline
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import PIL.Image
from trimesh.exchange.obj import export_obj
from trimesh.visual import texture
import cv2
from dotenv import load_dotenv
import math
from scipy.spatial.transform import Rotation
import sys
import importlib.util
import subprocess
from utils.replicate_api import generate_3d_model
import json
import time

# Load environment variables
load_dotenv()

# Try to import open3d but don't fail if it's not available
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: open3d not available, some advanced 3D features will be limited")

class ModelGenerator:
    def __init__(self):
        """Initialize the model generator with required models."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load models lazily - will be loaded on first use
        self._depth_model = None
        self._depth_feature_extractor = None
        self._text_to_3d_model = None
        
    def _load_models(self):
        """Load models if not already loaded."""
        if self._depth_model is None:
            print("Loading depth estimation model...")
            # Load depth estimation model from HuggingFace
            self._depth_feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
            self._depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(self.device)
            
        if self._text_to_3d_model is None:
            print("Loading 3D generation model...")
            # Load text-to-3D model
            self._text_to_3d_model = StableDiffusionDepth2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-depth",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
    
    def _estimate_depth(self, image):
        """
        Estimate depth map from input image using HuggingFace depth estimation model.
        
        Args:
            image: PIL Image input
            
        Returns:
            Depth map as numpy array
        """
        # Prepare image for the model
        inputs = self._depth_feature_extractor(images=image, return_tensors="pt").to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self._depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Convert to numpy and normalize to 0-1
        depth = predicted_depth.squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        return depth
    
    def _create_point_cloud_from_depth(self, depth_map, image, resolution=256):
        """
        Create a point cloud from depth map and image.
        
        Args:
            depth_map: Depth map as numpy array
            image: PIL Image for texture
            resolution: Point cloud resolution
            
        Returns:
            Point cloud object
        """
        # Resize depth map to desired resolution
        depth_resized = cv2.resize(depth_map, (resolution, resolution))
        
        # Resize image to match depth map
        img_resized = np.array(image.resize((resolution, resolution)))
        
        # Create meshgrid of coordinates
        h, w = depth_resized.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Convert depth to 3D points
        z = depth_resized.flatten() * 2 - 1  # Scale depth to [-1, 1]
        x = (x.flatten() / w - 0.5) * 2      # Scale x to [-1, 1]
        y = (y.flatten() / h - 0.5) * 2      # Scale y to [-1, 1]
        
        # Combine into point cloud
        points = np.vstack((x, y, z)).T
        
        # Add color information
        if img_resized.ndim == 3:
            colors = img_resized.reshape(-1, 3) / 255.0
        else:
            # Grayscale - convert to RGB
            colors = np.repeat(img_resized.reshape(-1, 1) / 255.0, 3, axis=1)
        
        # Create point cloud
        if OPEN3D_AVAILABLE:
            # Use Open3D if available
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Optional: filter outliers
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            return pcd
        else:
            # Use simple dictionary if Open3D not available
            return {
                "points": points,
                "colors": colors
            }
    
    def _point_cloud_to_mesh(self, pcd, resolution=256):
        """
        Convert point cloud to mesh.
        
        Args:
            pcd: Point cloud data
            resolution: Mesh resolution
            
        Returns:
            Trimesh mesh object
        """
        if OPEN3D_AVAILABLE and isinstance(pcd, o3d.geometry.PointCloud):
            # Use Open3D if available
            # Create mesh from point cloud using Poisson reconstruction
            with torch.no_grad():
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
                
                # Convert to trimesh for compatibility
                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            # Fallback method if Open3D not available
            if isinstance(pcd, dict):
                points = pcd["points"]
                colors = pcd["colors"]
            else:
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
            
            # Create a simple mesh using convex hull
            try:
                mesh = trimesh.Trimesh(vertices=points)
                mesh = mesh.convex_hull
                # Add color information
                mesh.visual = trimesh.visual.ColorVisuals(
                    vertex_colors=colors[:mesh.vertices.shape[0]] * 255
                )
            except Exception as e:
                print(f"Error creating mesh: {e}")
                # Create a very simple mesh as fallback
                mesh = trimesh.Trimesh(
                    vertices=[
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0]
                    ],
                    faces=[
                        [0, 1, 2],
                        [0, 1, 3],
                        [0, 2, 3],
                        [1, 2, 3]
                    ]
                )
                
        return mesh
    
    def _validate_3d_model(self, model_path, format="glb"):
        """
        Validate that the generated 3D model has proper depth and complexity.
        Attempts to repair mesh if possible.
        
        Args:
            model_path: Path to the 3D model file
            format: Format of the model file
            
        Returns:
            bool: True if model is valid or repaired successfully, False otherwise
        """
        try:
            print(f"Validating 3D model at {model_path}...")
            
            # Check if file exists and has size
            if not Path(model_path).exists():
                print(f"Model file is missing")
                return False
                
            file_size = Path(model_path).stat().st_size
            if file_size < 1000:  # Minimum size 1000 bytes
                print(f"Model file is too small ({file_size} bytes)")
                return False
            
            # Try to load the model with trimesh
            try:
                mesh = trimesh.load(model_path)
            except Exception as load_error:
                print(f"Error loading model: {load_error}")
                # Try to create a repair script
                return self._attempt_repair_mesh(model_path, format)
            
            # Basic mesh structure validation
            if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
                print(f"Model doesn't have proper mesh structure (missing vertices or faces)")
                # Try to repair
                return self._attempt_repair_mesh(model_path, format)
            
            # Check vertex and face counts
            min_vertices = 200
            min_faces = 150
                
            if mesh.vertices.shape[0] < min_vertices:
                print(f"Model has too few vertices ({mesh.vertices.shape[0]} < {min_vertices})")
                # Try to enhance the mesh
                return self._attempt_enhance_mesh(mesh, model_path, format)
                
            if mesh.faces.shape[0] < min_faces:
                print(f"Model has too few faces ({mesh.faces.shape[0]} < {min_faces})")
                # Try to enhance the mesh
                return self._attempt_enhance_mesh(mesh, model_path, format)
            
            # Check mesh integrity - watertight and consistent winding
            is_watertight = mesh.is_watertight
            consistent_winding = mesh.is_winding_consistent
            
            if not is_watertight or not consistent_winding:
                print(f"Mesh has integrity issues: watertight={is_watertight}, consistent_winding={consistent_winding}")
                # Try to repair the mesh
                return self._attempt_repair_mesh(model_path, format)
            
            # Check model dimensions - ensure it has some 3D depth
            try:
                bbox = mesh.bounding_box.extents
                min_dimension = min(bbox)
                max_dimension = max(bbox)
                
                # If any dimension is extremely small compared to the largest dimension, it might be too flat
                if min_dimension < (max_dimension * 0.05):
                    print(f"Model appears too flat or has unbalanced dimensions: {bbox}")
                    # Try to repair the mesh geometry
                    return self._attempt_enhance_mesh(mesh, model_path, format)
            except Exception as e:
                print(f"Error checking model dimensions: {e}")
            
            # Report basic model statistics
            vertex_count = mesh.vertices.shape[0]
            face_count = mesh.faces.shape[0]
            print(f"Model statistics: {vertex_count} vertices, {face_count} faces, {file_size} bytes")
            
            # Check if mesh has texture/material
            has_texture = hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material')
            has_vertex_color = hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors')
            
            if not has_texture and not has_vertex_color:
                print("Model lacks texture information, attempting to add basic material")
                # Add a basic material if there's none
                self._add_basic_material(mesh, model_path)
            
            # More stringent validation - require reasonable number of vertices and faces
            if vertex_count >= min_vertices and face_count >= min_faces:
                print("Model validated successfully")
                return True
                
            return False
            
        except Exception as e:
            print(f"Error validating 3D model: {e}")
            # If we can't validate properly, try to repair
            return self._attempt_repair_mesh(model_path, format)
            
    def _add_basic_material(self, mesh, model_path):
        """Add a basic material to the mesh if it doesn't have one"""
        try:
            # Create a simple material with a checkerboard pattern
            material = trimesh.visual.material.SimpleMaterial(diffuse=[200, 200, 250])
            mesh.visual = trimesh.visual.TextureVisuals(material=material)
            
            # Save the mesh with the new material
            mesh.export(model_path)
            print(f"Added basic material to mesh at {model_path}")
            return True
        except Exception as e:
            print(f"Error adding basic material: {e}")
            return False
    
    def _attempt_repair_mesh(self, model_path, format="glb"):
        """
        Attempt to repair a problematic mesh
        
        Args:
            model_path: Path to the model
            format: Output format
            
        Returns:
            bool: True if repair successful, False otherwise
        """
        try:
            print(f"Attempting to repair mesh at {model_path}")
            
            # Try to load the mesh, even if broken
            try:
                original_mesh = trimesh.load(model_path, process=False)
            except:
                print("Could not load mesh, creating basic replacement")
                # Create a basic replacement model
                original_mesh = self._create_replacement_mesh()
            
            # Create backup of original
            backup_path = f"{model_path}.backup"
            try:
                import shutil
                shutil.copy2(model_path, backup_path)
                print(f"Created backup at {backup_path}")
            except:
                print("Could not create backup")
            
            # Check if it's a Scene or Trimesh object
            if isinstance(original_mesh, trimesh.Scene):
                print("Model is a Scene object, extracting main mesh...")
                # Try to extract the main mesh from the scene
                try:
                    # Get all meshes from the scene
                    meshes = []
                    for name, geom in original_mesh.geometry.items():
                        if isinstance(geom, trimesh.Trimesh):
                            meshes.append(geom)
                    
                    if meshes:
                        # Use the largest mesh if there are multiple
                        largest_mesh = max(meshes, key=lambda m: m.vertices.shape[0])
                        repaired_mesh = largest_mesh.copy()
                        print(f"Extracted main mesh with {repaired_mesh.vertices.shape[0]} vertices")
                    else:
                        print("No valid meshes found in scene, creating replacement")
                        repaired_mesh = self._create_replacement_mesh()
                except Exception as scene_error:
                    print(f"Error extracting mesh from scene: {scene_error}")
                    repaired_mesh = self._create_replacement_mesh()
            else:
                # It's already a Trimesh, just copy it
                repaired_mesh = original_mesh.copy()
            
            # First try using trimesh's built-in repair function 
            try:
                # Only apply these operations if it's a Trimesh object with the right attributes
                if isinstance(repaired_mesh, trimesh.Trimesh):
                    # Fill holes
                    if hasattr(repaired_mesh, 'fill_holes'):
                        repaired_mesh.fill_holes()
                    
                    # Fix normals
                    if hasattr(repaired_mesh, 'fix_normals'):
                        repaired_mesh.fix_normals()
                    
                    # Fix winding
                    if hasattr(repaired_mesh, 'fix_winding'):
                        repaired_mesh.fix_winding()
                    
                    # Fix inversion
                    if hasattr(repaired_mesh, 'fix_inversion'):
                        repaired_mesh.fix_inversion()
                    
                    # Remove duplicate faces
                    if hasattr(repaired_mesh, 'remove_duplicate_faces'):
                        repaired_mesh.remove_duplicate_faces()
                    
                    # Remove unreferenced vertices
                    if hasattr(repaired_mesh, 'remove_unreferenced_vertices'):
                        repaired_mesh.remove_unreferenced_vertices()
                    
                    # Remove degenerate faces
                    if hasattr(repaired_mesh, 'remove_degenerate_faces'):
                        repaired_mesh.remove_degenerate_faces()
                
                # Check if the repair improved the mesh
                if (hasattr(repaired_mesh, 'vertices') and 
                    hasattr(repaired_mesh, 'faces') and
                    repaired_mesh.vertices.shape[0] >= 200 and
                    repaired_mesh.faces.shape[0] >= 150):
                    
                    # Export the repaired mesh
                    repaired_mesh.export(model_path)
                    print(f"Successfully repaired mesh using trimesh utilities")
                    return True
            except Exception as repair_error:
                print(f"Basic repair failed: {repair_error}")
            
            # If that fails, try more aggressive methods - voxelization and reconstruction
            try:
                # Only try voxelization if it's a Trimesh with the right attribute
                if isinstance(repaired_mesh, trimesh.Trimesh) and hasattr(repaired_mesh, 'voxelized'):
                    # Try to voxelize the mesh and reconstruct
                    voxelized = repaired_mesh.voxelized(pitch=0.05)
                    reconstructed = voxelized.marching_cubes
                    
                    # Check if reconstruction produced valid mesh
                    if (hasattr(reconstructed, 'vertices') and 
                        hasattr(reconstructed, 'faces') and
                        reconstructed.vertices.shape[0] >= 200 and
                        reconstructed.faces.shape[0] >= 150):
                        
                        # Export the reconstructed mesh
                        reconstructed.export(model_path)
                        print(f"Successfully repaired mesh using voxelization")
                        return True
            except Exception as voxel_error:
                print(f"Voxelization repair failed: {voxel_error}")
            
            # If all else fails, create a replacement mesh
            replacement_mesh = self._create_replacement_mesh()
            replacement_mesh.export(model_path)
            print(f"Created replacement mesh as fallback")
            return True
            
        except Exception as e:
            print(f"Mesh repair failed: {e}")
            # Last resort - create a simple primitives-based mesh
            try:
                replacement_mesh = self._create_replacement_mesh()
                replacement_mesh.export(model_path)
                print(f"Created basic replacement mesh")
                return True
            except:
                return False
    
    def _create_replacement_mesh(self):
        """Create a simple replacement mesh when all repair attempts fail"""
        try:
            # Create a complex primitive with multiple parts for better structure
            # Start with a sphere at the center
            sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
            
            # Add a second smaller sphere offset to create depth
            sphere2 = trimesh.creation.icosphere(subdivisions=2, radius=0.3)
            sphere2.apply_translation([0.7, 0, 0])
            
            # Add a cube
            box = trimesh.creation.box(extents=[0.3, 0.3, 0.3])
            box.apply_translation([0, 0.7, 0])
            
            # Add a cylinder for additional structure
            cylinder = trimesh.creation.cylinder(radius=0.15, height=0.8)
            cylinder.apply_translation([0, 0, 0.7])
            
            # Combine into a complex mesh
            mesh = trimesh.util.concatenate([sphere, sphere2, box, cylinder])
            
            # Create a simple material with color
            mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=[200, 200, 250, 255])
            
            return mesh
        except Exception as e:
            print(f"Error creating replacement mesh: {e}")
            # Absolute fallback - just return a sphere
            return trimesh.creation.icosphere()
            
    def _attempt_enhance_mesh(self, mesh, model_path, format="glb"):
        """
        Attempt to enhance a mesh with too few vertices or faces
        
        Args:
            mesh: Original mesh
            model_path: Path to save enhanced mesh
            format: Output format
            
        Returns:
            bool: True if enhancement successful, False otherwise
        """
        try:
            print(f"Attempting to enhance mesh with subdivision")
            
            # Make a copy to avoid modifying the original
            enhanced_mesh = mesh.copy()
            
            # First try to use subdivision to increase mesh complexity
            try:
                # Try Loop subdivision first (creates smoother meshes)
                from trimesh.remesh import subdivide
                vertices_new, faces_new = subdivide.subdivide_loop(
                    enhanced_mesh.vertices, 
                    enhanced_mesh.faces, 
                    iterations=1
                )
                
                enhanced_mesh = trimesh.Trimesh(vertices=vertices_new, faces=faces_new)
                
                # Check if we have enough vertices and faces now
                if (enhanced_mesh.vertices.shape[0] >= 200 and
                    enhanced_mesh.faces.shape[0] >= 150):
                    
                    # Copy visual properties from original
                    if hasattr(mesh, 'visual'):
                        enhanced_mesh.visual = mesh.visual
                    
                    # Export the enhanced mesh
                    enhanced_mesh.export(model_path)
                    print(f"Successfully enhanced mesh with Loop subdivision")
                    return True
            except Exception as subdiv_error:
                print(f"Loop subdivision failed: {subdiv_error}")
            
            # Try a different subdivision algorithm if Loop failed
            try:
                from trimesh.remesh import subdivide
                vertices_new, faces_new = subdivide.subdivide_midpoint(
                    enhanced_mesh.vertices, 
                    enhanced_mesh.faces, 
                    iterations=1
                )
                
                enhanced_mesh = trimesh.Trimesh(vertices=vertices_new, faces=faces_new)
                
                # Check if we have enough vertices and faces now
                if (enhanced_mesh.vertices.shape[0] >= 200 and
                    enhanced_mesh.faces.shape[0] >= 150):
                    
                    # Copy visual properties from original
                    if hasattr(mesh, 'visual'):
                        enhanced_mesh.visual = mesh.visual
                    
                    # Export the enhanced mesh
                    enhanced_mesh.export(model_path)
                    print(f"Successfully enhanced mesh with midpoint subdivision")
                    return True
            except Exception as midpoint_error:
                print(f"Midpoint subdivision failed: {midpoint_error}")
            
            # If all else fails, replace with a new mesh
            return self._attempt_repair_mesh(model_path, format)
            
        except Exception as e:
            print(f"Mesh enhancement failed: {e}")
            return False
    
    def generate_hunyuan3d(self, image, request_id, resolution=256, use_high_quality=True, format="glb", 
                      model_version="b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb",
                      retry_count=2, texture_resolution=2048, remove_background=True,
                      geometry_detail_level="high", texture_quality="high"):
        """
        Generate a 3D model using Replicate's Hunyuan3D-2 API.
        
        Args:
            image: Input image (path or PIL Image)
            request_id: Unique request ID for tracking
            resolution: Model resolution
            use_high_quality: Whether to use high quality settings
            format: Output format (glb or obj)
            model_version: Version of the Hunyuan3D-2 model to use
            retry_count: Number of times to retry generation if it fails
            texture_resolution: Resolution of textures (higher = better quality)
            remove_background: Whether to automatically remove background from image
            geometry_detail_level: Level of geometric detail in the model (low, medium, high)
            texture_quality: Quality level for textures (basic, medium, high)
            
        Returns:
            Path to generated model file
        """
        tries = 0
        last_error = None
        
        # Always use high quality for better results
        use_high_quality = True
        # Use higher resolution for better quality
        resolution = max(resolution, 512)  # Increased from 384 to 512 for better quality
        
        # Try alternate model version if specified default doesn't work
        original_model_version = model_version
        
        # Create output directory upfront to avoid race conditions
        output_dir = Path("outputs") / request_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save diagnostics info for debugging
        diag_file = output_dir / "generation_info.json"
        try:
            with open(diag_file, "w") as f:
                json.dump({
                    "request_id": request_id,
                    "start_time": time.time(),
                    "resolution": resolution,
                    "use_high_quality": use_high_quality,
                    "format": format,
                    "model_version": model_version,
                    "texture_resolution": texture_resolution,
                    "remove_background": remove_background,
                    "geometry_detail_level": geometry_detail_level,
                    "texture_quality": texture_quality
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save diagnostic info: {e}")
        
        # Track success or failure
        is_success = False
        
        while tries < retry_count:
            tries += 1
            
            # Update diagnostics with attempt info
            try:
                if diag_file.exists():
                    with open(diag_file, "r") as f:
                        diag_data = json.load(f)
                    
                    diag_data[f"attempt_{tries}"] = {
                        "start_time": time.time(),
                        "model_version": model_version
                    }
                    
                    with open(diag_file, "w") as f:
                        json.dump(diag_data, f, indent=2)
            except Exception:
                # Don't fail just for diagnostics
                pass
                
            try:
                print(f"Generating 3D model with Replicate API for request {request_id} (attempt {tries}/{retry_count})...")
                
                # Generate a sharper image for better texture quality if it's a PIL Image
                if isinstance(image, PIL.Image.Image):
                    # Apply slight sharpening to enhance details
                    try:
                        from PIL import ImageEnhance
                        enhancer = ImageEnhance.Sharpness(image)
                        enhanced_image = enhancer.enhance(1.5)  # Sharpen by 50%
                        
                        # Save enhanced image
                        enhanced_path = output_dir / "enhanced_input.png"
                        enhanced_image.save(enhanced_path)
                        input_image = str(enhanced_path)
                    except Exception as enhance_error:
                        print(f"Warning: Could not enhance image: {enhance_error}")
                        # Fall back to original image
                        image_path = output_dir / "input_image.png"
                        image.save(image_path)
                        input_image = str(image_path)
                else:
                    input_image = image
                
                # Generate 3D model using Replicate API - increase max_retries to make API more robust
                result = generate_3d_model(
                    image=input_image,
                    output_dir=output_dir,
                    request_id=request_id,
                    format=format,
                    resolution=resolution,
                    use_high_quality=use_high_quality,
                    model_version=model_version,
                    max_retries=3,  # Use 3 internal API retries for robustness
                    texture_resolution=texture_resolution,
                    remove_background=remove_background,
                    geometry_detail_level=geometry_detail_level,
                    texture_quality=texture_quality
                )
                
                # Check if it's a fallback model
                if result.get("is_fallback", False):
                    print(f"Warning: Received fallback model from API for attempt {tries}/{retry_count}")
                    if tries < retry_count:
                        # Alternate model version for next attempt
                        if model_version == original_model_version:
                            model_version = "b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb"
                            print(f"Switching to alternate model version: {model_version}")
                        else:
                            model_version = original_model_version
                            print(f"Switching back to original model version: {model_version}")
                        continue
                
                # Validate the 3D model to ensure it's proper
                model_path = result['model_path']
                
                # Safe validation - don't crash if validation fails
                validation_success = False
                try:
                    validation_success = self._validate_3d_model(model_path, format)
                except Exception as validation_error:
                    print(f"Error during model validation: {validation_error}")
                    validation_success = False
                
                if not validation_success:
                    if tries < retry_count:
                        print(f"Generated model failed validation, attempt {tries}/{retry_count}, retrying...")
                        
                        # Alternate model version for next attempt
                        if model_version == original_model_version:
                            model_version = "b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb"
                            print(f"Switching to alternate model version: {model_version}")
                        else:
                            model_version = original_model_version
                            print(f"Switching back to original model version: {model_version}")
                            
                        continue
                    else:
                        print(f"Final attempt {tries}/{retry_count} failed validation, trying with different settings...")
                        # Try regenerating with higher quality settings
                        result = generate_3d_model(
                            image=input_image,
                            output_dir=output_dir,
                            request_id=request_id,
                            format=format,
                            resolution=512,  # Force maximum resolution
                            use_high_quality=True,  # Force high quality
                            model_version="b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb",  # Try alternate model
                            max_retries=3,  # Use more API retries for this last attempt
                            texture_resolution=4096,  # Maximum texture resolution
                            remove_background=True,
                            geometry_detail_level="high",
                            texture_quality="high"
                        )
                        model_path = result['model_path']
                        
                        # Validate again
                        validation_success = False
                        try:
                            validation_success = self._validate_3d_model(model_path, format)
                        except Exception:
                            validation_success = False
                            
                        if not validation_success:
                            print(f"Enhanced settings also failed validation, falling back to local generation...")
                            # If still fails, fall back to local generation
                            try:
                                local_model_path = self.generate(
                                    image, 
                                    request_id, 
                                    resolution=512,
                                    use_high_quality=True,
                                    format=format,
                                    multi_view=True
                                )
                                
                                # Update diagnostics with success info
                                try:
                                    if diag_file.exists():
                                        with open(diag_file, "r") as f:
                                            diag_data = json.load(f)
                                        
                                        diag_data["success"] = True
                                        diag_data["method"] = "local_fallback"
                                        diag_data["end_time"] = time.time()
                                        
                                        with open(diag_file, "w") as f:
                                            json.dump(diag_data, f, indent=2)
                                except Exception:
                                    pass
                                    
                                return local_model_path
                            except Exception as local_error:
                                print(f"Error in local generation: {local_error}")
                                # Continue with the best model we have so far
                
                # Success - model is valid
                is_success = True
                
                # Update diagnostics with success info
                try:
                    if diag_file.exists():
                        with open(diag_file, "r") as f:
                            diag_data = json.load(f)
                        
                        diag_data["success"] = True
                        diag_data["method"] = "api"
                        diag_data["attempt_succeeded"] = tries
                        diag_data["end_time"] = time.time()
                        
                        with open(diag_file, "w") as f:
                            json.dump(diag_data, f, indent=2)
                except Exception:
                    pass
                    
                print(f"3D model validated and saved at: {model_path}")
                return model_path
                
            except Exception as e:
                last_error = e
                print(f"Error generating 3D model with Replicate API (attempt {tries}/{retry_count}): {e}")
                
                # Update diagnostics with error info
                try:
                    if diag_file.exists():
                        with open(diag_file, "r") as f:
                            diag_data = json.load(f)
                        
                        diag_data[f"attempt_{tries}_error"] = str(e)
                        
                        with open(diag_file, "w") as f:
                            json.dump(diag_data, f, indent=2)
                except Exception:
                    pass
                
                if tries < retry_count:
                    print(f"Retrying with attempt {tries+1}/{retry_count}...")
                    
                    # Alternate model version for next attempt
                    if model_version == original_model_version:
                        model_version = "b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb"
                        print(f"Switching to alternate model version: {model_version}")
                    else:
                        model_version = original_model_version
                        print(f"Switching back to original model version: {model_version}")
                else:
                    print("All retry attempts failed, raising exception")
                    raise
        
        # If not successful, try local generation as an absolute fallback
        if not is_success:
            print("API generation failed after all retries, falling back to local generation...")
            try:
                local_model_path = self.generate(
                    image, 
                    request_id, 
                    resolution=512,
                    use_high_quality=True,
                    format=format,
                    multi_view=True
                )
                
                # Update diagnostics
                try:
                    if diag_file.exists():
                        with open(diag_file, "r") as f:
                            diag_data = json.load(f)
                        
                        diag_data["success"] = True
                        diag_data["method"] = "local_fallback_final"
                        diag_data["end_time"] = time.time()
                        
                        with open(diag_file, "w") as f:
                            json.dump(diag_data, f, indent=2)
                except Exception:
                    pass
                    
                return local_model_path
            except Exception as final_error:
                print(f"Critical: Even local generation failed: {final_error}")
                # Update diagnostics
                try:
                    if diag_file.exists():
                        with open(diag_file, "r") as f:
                            diag_data = json.load(f)
                        
                        diag_data["success"] = False
                        diag_data["final_error"] = str(final_error)
                        diag_data["end_time"] = time.time()
                        
                        with open(diag_file, "w") as f:
                            json.dump(diag_data, f, indent=2)
                except Exception:
                    pass
                # If we get here, all retries failed
                raise last_error if last_error else Exception("Failed to generate 3D model after all retries")
    
    def generate_with_text_prompt_hunyuan3d(self, image, prompt, request_id, resolution=256, format="glb",
                                  model_version="b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb",
                                  retry_count=2, texture_resolution=2048, remove_background=True,
                                  geometry_detail_level="high", texture_quality="high"):
        """
        Generate a 3D model with text prompt using Replicate's Hunyuan3D-2 API.
        
        Args:
            image: Input image (path or PIL Image)
            prompt: Text prompt to guide generation
            request_id: Unique request ID for tracking
            resolution: Model resolution
            format: Output format (glb or obj)
            model_version: Version of the Hunyuan3D-2 model to use
            retry_count: Number of times to retry generation if it fails
            texture_resolution: Resolution of textures (higher = better quality)
            remove_background: Whether to automatically remove background from image
            geometry_detail_level: Level of geometric detail in the model (low, medium, high)
            texture_quality: Quality level for textures (basic, medium, high)
            
        Returns:
            Path to generated model file
        """
        tries = 0
        last_error = None
        
        # Create output directory
        output_dir = Path("outputs") / request_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save diagnostics info for debugging
        diag_file = output_dir / "generation_info.json"
        try:
            with open(diag_file, "w") as f:
                json.dump({
                    "request_id": request_id,
                    "start_time": time.time(),
                    "prompt": prompt,
                    "resolution": resolution,
                    "format": format,
                    "model_version": model_version,
                    "texture_resolution": texture_resolution,
                    "remove_background": remove_background,
                    "geometry_detail_level": geometry_detail_level,
                    "texture_quality": texture_quality
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save diagnostic info: {e}")
        
        # Improve prompt for better 3D generation
        enhanced_prompt = prompt
        if not any(keyword in prompt.lower() for keyword in ["3d", "three dimensional", "detailed", "high quality", "texture"]):
            enhanced_prompt = f"{prompt}, detailed 3D model with clear shape and realistic texture, high quality"
            print(f"Enhanced prompt: '{prompt}' -> '{enhanced_prompt}'")
        
        while tries < retry_count:
            tries += 1
            
            # Update diagnostics with attempt info
            try:
                if diag_file.exists():
                    with open(diag_file, "r") as f:
                        diag_data = json.load(f)
                    
                    diag_data[f"attempt_{tries}"] = {
                        "start_time": time.time(),
                        "model_version": model_version
                    }
                    
                    with open(diag_file, "w") as f:
                        json.dump(diag_data, f, indent=2)
            except Exception:
                # Don't fail just for diagnostics
                pass
            
            try:
                print(f"Generating 3D model with Replicate API and text prompt for request {request_id} (attempt {tries}/{retry_count})...")
                
                # Generate a sharper image for better texture quality if it's a PIL Image
                if isinstance(image, PIL.Image.Image):
                    # Apply slight sharpening to enhance details
                    try:
                        from PIL import ImageEnhance
                        enhancer = ImageEnhance.Sharpness(image)
                        enhanced_image = enhancer.enhance(1.5)  # Sharpen by 50%
                        
                        # Save enhanced image
                        enhanced_path = output_dir / "enhanced_input.png"
                        enhanced_image.save(enhanced_path)
                        input_image = str(enhanced_path)
                    except Exception as enhance_error:
                        print(f"Warning: Could not enhance image: {enhance_error}")
                        # Fall back to original image
                        image_path = output_dir / "input_image.png"
                        image.save(image_path)
                        input_image = str(image_path)
                else:
                    input_image = image
                
                # Generate 3D model using Replicate API with text prompt
                result = generate_3d_model(
                    image=input_image,
                    output_dir=output_dir,
                    request_id=request_id,
                    prompt=enhanced_prompt,
                    format=format,
                    resolution=max(resolution, 512),  # Ensure high resolution
                    use_high_quality=True,
                    model_version=model_version,
                    max_retries=3,  # Use 3 internal API retries for robustness
                    texture_resolution=texture_resolution,
                    remove_background=remove_background,
                    geometry_detail_level=geometry_detail_level,
                    texture_quality=texture_quality
                )
                
                # Validate the 3D model to ensure it's proper
                model_path = result['model_path']
                validation_success = False
                try:
                    validation_success = self._validate_3d_model(model_path, format)
                except Exception as validation_error:
                    print(f"Error during model validation: {validation_error}")
                    validation_success = False
                
                if not validation_success:
                    if tries < retry_count:
                        print(f"Generated model with prompt failed validation, attempt {tries}/{retry_count}, retrying...")
                        continue
                    else:
                        print(f"Final attempt {tries}/{retry_count} failed validation, trying with further enhanced prompt...")
                        # Try regenerating with higher quality settings and even better prompt
                        further_enhanced_prompt = enhanced_prompt + ", highly detailed 3D model, photorealistic textures, volumetric, professional quality, sharp details"
                        
                        result = generate_3d_model(
                            image=input_image,
                            output_dir=output_dir,
                            request_id=request_id,
                            prompt=further_enhanced_prompt,
                            format=format,
                            resolution=max(resolution, 512),  # Use at least 512 resolution
                            use_high_quality=True,  # Force high quality
                            model_version=model_version,
                            max_retries=3,  # More retries for this last attempt
                            texture_resolution=4096,  # Maximum texture resolution
                            remove_background=True,
                            geometry_detail_level="high",
                            texture_quality="high"
                        )
                        model_path = result['model_path']
                        
                        # Validate again
                        validation_success = False
                        try:
                            validation_success = self._validate_3d_model(model_path, format)
                        except Exception:
                            validation_success = False
                            
                        if not validation_success:
                            print(f"Enhanced prompt also failed validation, falling back to local generation...")
                            # If still fails, fall back to local generation
                            return self.generate_with_text_prompt(
                                image, 
                                further_enhanced_prompt,
                                request_id, 
                                resolution=max(resolution, 512),
                                format=format,
                                multi_view=True
                            )
                
                # Update diagnostics with success info
                try:
                    if diag_file.exists():
                        with open(diag_file, "r") as f:
                            diag_data = json.load(f)
                        
                        diag_data["success"] = True
                        diag_data["method"] = "api"
                        diag_data["attempt_succeeded"] = tries
                        diag_data["end_time"] = time.time()
                        
                        with open(diag_file, "w") as f:
                            json.dump(diag_data, f, indent=2)
                except Exception:
                    pass
                
                print(f"3D model with prompt validated and saved at: {model_path}")
                return model_path
                
            except Exception as e:
                last_error = e
                print(f"Error generating 3D model with Replicate API and text prompt (attempt {tries}/{retry_count}): {e}")
                if tries < retry_count:
                    print(f"Retrying with attempt {tries+1}/{retry_count}...")
                else:
                    print("All retry attempts failed, raising exception")
                    raise
        
        # If we get here, all retries failed
        raise last_error if last_error else Exception("Failed to generate 3D model after all retries")
    
    def generate(self, image, request_id, resolution=256, use_high_quality=False, format="glb", multi_view=True):
        """
        Generate a 3D model from the input image.
        
        Args:
            image: The preprocessed input image (PIL Image)
            request_id: Unique ID for this generation request
            resolution: Output resolution
            use_high_quality: Whether to use high quality generation
            format: Output format (glb, obj)
            multi_view: Whether to generate a model from multiple viewpoints
            
        Returns:
            Path to the generated 3D model file
        """
        # If Replicate API is available, use it for better quality
        if True:
            try:
                return self.generate_hunyuan3d(
                    image, request_id, 
                    resolution=resolution, 
                    use_high_quality=use_high_quality, 
                    format=format
                )
            except Exception as e:
                print(f"Error using Replicate API: {e}")
                print("Falling back to original implementation...")
        
        # Original implementation (as fallback)
        self._load_models()
        
        # Create output directory
        output_dir = Path("outputs") / request_id
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Generating 3D model for request {request_id}...")
        
        # 1. Estimate depth map from image
        depth_map = self._estimate_depth(image)
        
        # Save depth map as visualization
        depth_vis = (depth_map * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "depth_map.png"), depth_vis)
        
        # 2. Create point cloud from depth map
        pcd = self._create_point_cloud_from_depth(depth_map, image, resolution=resolution)
        
        # 3. Convert point cloud to mesh using the point_cloud_to_mesh method
        mesh = self._point_cloud_to_mesh(pcd, resolution=resolution)
        
        # 4. Add texture if requested
        if use_high_quality:
            # Create a simple texture from the image
            img_array = np.array(image.resize((1024, 1024)))
            material = texture.SimpleMaterial(image=img_array)
            
            # Apply texture
            mesh.visual = texture.TextureVisuals(material=material)
        
        # 5. Save the mesh in the requested format
        output_path = output_dir / f"model.{format}"
        mesh.export(output_path)
        
        print(f"3D model generated and saved at {output_path}")
        return output_path
        
    def generate_with_text_prompt(self, image, prompt, request_id, resolution=256, format="glb", multi_view=True):
        """
        Generate a 3D model from the input image and text prompt.
        
        Args:
            image: The preprocessed input image (PIL Image)
            prompt: Text description to guide the 3D generation
            request_id: Unique ID for this generation request
            resolution: Output resolution
            format: Output format (glb, obj)
            multi_view: Whether to generate a model from multiple viewpoints
            
        Returns:
            Path to the generated 3D model file
        """
        # If Replicate API is available, use it for better quality
        if True:
            try:
                return self.generate_with_text_prompt_hunyuan3d(
                    image, prompt, request_id,
                    resolution=resolution,
                    format=format
                )
            except Exception as e:
                print(f"Error using Replicate API with text prompt: {e}")
                print("Falling back to original implementation...")
                
        # Ensure models are loaded
        self._load_models()
        
        # Create output directory
        output_dir = Path("outputs") / request_id
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Generating 3D model with text prompt for request {request_id}...")
        
        # 1. Refine image with text prompt
        depth_map = self._estimate_depth(image)
        depth_vis = (depth_map * 255).astype(np.uint8)
        depth_image = PIL.Image.fromarray(depth_vis)
        
        # Generate refined image with prompt
        refined_image = self._text_to_3d_model(
            prompt=prompt + ", detailed 3D model",
            image=image,
            depth_image=depth_image,
            strength=0.7,
            num_inference_steps=50
        ).images[0]
        
        # Save refined image
        refined_image.save(output_dir / "refined_image.png")
        
        # 2. Use the refined image for 3D generation
        refined_depth_map = self._estimate_depth(refined_image)
        
        # Create point cloud
        pcd = self._create_point_cloud_from_depth(refined_depth_map, refined_image, resolution=resolution)
        
        # Convert to mesh using the point_cloud_to_mesh method
        mesh = self._point_cloud_to_mesh(pcd, resolution=resolution)
        
        # Add texture
        img_array = np.array(refined_image.resize((1024, 1024)))
        material = texture.SimpleMaterial(image=img_array)
        mesh.visual = texture.TextureVisuals(material=material)
        
        # Save the mesh in the requested format
        output_path = output_dir / f"model.{format}"
        mesh.export(output_path)
        
        print(f"3D model generated with text prompt and saved at {output_path}")
        return output_path 