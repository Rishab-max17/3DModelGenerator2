import torch
from pathlib import Path
import numpy as np
import trimesh
import os
from huggingface_hub import hf_hub_download, login, snapshot_download
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

# Load environment variables
load_dotenv()

# Get API token from environment
hf_token = os.getenv("HF_API_TOKEN")
if hf_token:
    # Login to Hugging Face with token from environment
    login(token=hf_token)
else:
    print("Warning: HF_API_TOKEN not found in environment variables")

# Try to import open3d but don't fail if it's not available
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: open3d not available, some advanced 3D features will be limited")

# Check if Hunyuan3D repo is available locally
HUNYUAN3D_AVAILABLE = False
HUNYUAN3D_REPO_PATH = Path("hunyuan3d")

def setup_hunyuan3d():
    """Download and set up Hunyuan3D-2 from HuggingFace"""
    global HUNYUAN3D_AVAILABLE, HUNYUAN3D_REPO_PATH
    
    try:
        print("Setting up Hunyuan3D-2 from Hugging Face...")
        
        # Download the repository from Hugging Face if not already downloaded
        if not HUNYUAN3D_REPO_PATH.exists():
            # Clone the repository
            subprocess.run(
                ["git", "clone", "https://huggingface.co/tencent/Hunyuan3D-2", str(HUNYUAN3D_REPO_PATH)],
                check=True
            )
            
        # Add the repo to Python path
        sys.path.append(str(HUNYUAN3D_REPO_PATH))
        
        # Download model weights
        snapshot_download(
            repo_id="tencent/Hunyuan3D-2",
            local_dir=str(HUNYUAN3D_REPO_PATH),
            token=hf_token
        )
        
        # Try to import to verify setup worked
        spec = importlib.util.spec_from_file_location(
            "hy3dgen", 
            str(HUNYUAN3D_REPO_PATH / "minimal_demo.py")
        )
        if spec:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            HUNYUAN3D_AVAILABLE = True
            print("Successfully set up Hunyuan3D-2")
        else:
            print("Failed to load Hunyuan3D-2 module")
            HUNYUAN3D_AVAILABLE = False
            
    except Exception as e:
        print(f"Error setting up Hunyuan3D-2: {e}")
        HUNYUAN3D_AVAILABLE = False

class ModelGenerator:
    def __init__(self):
        """Initialize the model generator with required models."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Check if Hunyuan3D is available
        if not HUNYUAN3D_AVAILABLE:
            setup_hunyuan3d()
        
        # Load models lazily - will be loaded on first use
        self._depth_model = None
        self._depth_feature_extractor = None
        self._text_to_3d_model = None
        self._hunyuan_shape_model = None
        self._hunyuan_texture_model = None
        
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
    
    def _load_hunyuan3d_models(self):
        """Load Hunyuan3D models if not already loaded."""
        if HUNYUAN3D_AVAILABLE:
            if self._hunyuan_shape_model is None or self._hunyuan_texture_model is None:
                try:
                    print("Loading Hunyuan3D-2 models...")
                    # Import the modules from the cloned repository
                    sys.path.append(str(HUNYUAN3D_REPO_PATH))
                    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
                    from hy3dgen.texgen import Hunyuan3DPaintPipeline
                    
                    # Load the shape generation model
                    self._hunyuan_shape_model = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                        str(HUNYUAN3D_REPO_PATH),
                        use_safetensors=True,
                        device=self.device
                    )
                    
                    # Load the texture model
                    self._hunyuan_texture_model = Hunyuan3DPaintPipeline.from_pretrained(
                        str(HUNYUAN3D_REPO_PATH),
                        use_safetensors=True,
                        device=self.device
                    )
                except Exception as e:
                    print(f"Error loading Hunyuan3D-2 models: {e}")
                    raise ImportError(f"Failed to load Hunyuan3D-2 models: {e}")
        else:
            raise ImportError("Hunyuan3D-2 is not available. Please check setup.")
    
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
    
    def _create_point_cloud_from_depth(self, depth_map, image=None, resolution=256):
        """
        Create a colored point cloud from a depth map and optional color image.
        
        Args:
            depth_map: Depth map as numpy array
            image: Optional color image (PIL Image)
            resolution: Output resolution
            
        Returns:
            Open3D point cloud or trimesh point cloud
        """
        # Resize depth map to requested resolution
        depth_map = cv2.resize(depth_map, (resolution, resolution))
        
        # Get color information if available
        if image is not None:
            # Resize image to match depth map
            image = image.resize((resolution, resolution))
            color_data = np.array(image)
        else:
            # Use grayscale if no image
            color_data = np.zeros((resolution, resolution, 3), dtype=np.uint8)
            color_data[:, :, 0] = depth_map * 255
            color_data[:, :, 1] = depth_map * 255
            color_data[:, :, 2] = depth_map * 255
            
        # Create coordinate grid
        h, w = depth_map.shape
        y, x = np.mgrid[0:h, 0:w]
        
        # Scale coordinates to [-0.5, 0.5]
        x = (x / w) - 0.5
        y = (y / h) - 0.5
        z = depth_map * 0.5  # Depth scale factor (increased for better depth)
        
        # Create points
        points = np.zeros((h*w, 3))
        points[:, 0] = x.flatten()
        points[:, 1] = -y.flatten()  # Flip Y to match OpenGL convention
        points[:, 2] = z.flatten()
        
        # Create colors
        colors = color_data.reshape(-1, 3) / 255.0
        
        if OPEN3D_AVAILABLE:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            return pcd
        else:
            # Create trimesh point cloud
            return trimesh.PointCloud(points, colors=colors)
    
    def _transform_point_cloud(self, pcd, angle_degrees, translation=None):
        """
        Transform a point cloud with rotation and optional translation.
        
        Args:
            pcd: Point cloud (Open3D or trimesh)
            angle_degrees: Rotation angle in degrees (around Y axis)
            translation: Optional translation vector
            
        Returns:
            Transformed point cloud
        """
        # Convert angle to radians
        angle_rad = math.radians(angle_degrees)
        
        # Create rotation matrix (around Y axis)
        r = Rotation.from_euler('y', angle_degrees, degrees=True)
        rotation_matrix = r.as_matrix()
        
        if OPEN3D_AVAILABLE and isinstance(pcd, o3d.geometry.PointCloud):
            # Apply rotation to Open3D point cloud
            rotation_matrix_4x4 = np.eye(4)
            rotation_matrix_4x4[:3, :3] = rotation_matrix
            
            # Add translation if provided
            if translation is not None:
                rotation_matrix_4x4[:3, 3] = translation
                
            pcd_transformed = pcd.transform(rotation_matrix_4x4)
            return pcd_transformed
        else:
            # Apply rotation to trimesh point cloud
            points = pcd.vertices
            transformed_points = np.dot(points, rotation_matrix.T)
            
            # Apply translation if provided
            if translation is not None:
                transformed_points += translation
                
            # Create new point cloud
            pcd_transformed = trimesh.PointCloud(transformed_points, colors=pcd.colors)
            return pcd_transformed
    
    def _integrate_point_clouds(self, point_clouds, angles, voxel_size=0.01):
        """
        Integrate multiple point clouds into a single point cloud.
        
        Args:
            point_clouds: List of point clouds
            angles: List of rotation angles for each point cloud
            voxel_size: Voxel size for downsampling
            
        Returns:
            Integrated point cloud
        """
        if len(point_clouds) == 1:
            return point_clouds[0]
            
        # Transform point clouds based on their angles
        transformed_clouds = []
        for i, (pcd, angle) in enumerate(zip(point_clouds, angles)):
            # Front view doesn't need rotation
            if angle == 0:
                transformed_clouds.append(pcd)
            else:
                # Define translation based on the angle to ensure views
                # are properly positioned around the object center
                if angle == 90:  # Right side view
                    translation = [0.5, 0, 0]
                elif angle == 180:  # Back view
                    translation = [0, 0, -0.5]
                elif angle == 270:  # Left side view
                    translation = [-0.5, 0, 0]
                else:
                    translation = None
                    
                transformed_pcd = self._transform_point_cloud(pcd, angle, translation)
                transformed_clouds.append(transformed_pcd)
        
        if OPEN3D_AVAILABLE:
            # Combine all point clouds
            combined_pcd = o3d.geometry.PointCloud()
            for pcd in transformed_clouds:
                combined_pcd += pcd
            
            # Voxel downsampling to remove duplicate points
            combined_pcd = combined_pcd.voxel_down_sample(voxel_size)
            
            # Remove outliers
            cl, ind = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            combined_pcd = combined_pcd.select_by_index(ind)
            
            return combined_pcd
        else:
            # For trimesh, combine points and colors
            all_points = []
            all_colors = []
            
            for pcd in transformed_clouds:
                all_points.append(pcd.vertices)
                all_colors.append(pcd.colors)
                
            combined_points = np.vstack(all_points)
            combined_colors = np.vstack(all_colors)
            
            # Simple downsampling by random selection
            if len(combined_points) > 100000:
                indices = np.random.choice(len(combined_points), 100000, replace=False)
                combined_points = combined_points[indices]
                combined_colors = combined_colors[indices]
                
            return trimesh.PointCloud(combined_points, colors=combined_colors)
    
    def _point_cloud_to_mesh(self, point_cloud, resolution=256):
        """
        Convert a point cloud to a mesh.
        
        Args:
            point_cloud: Point cloud (Open3D or trimesh)
            resolution: Target resolution
            
        Returns:
            Trimesh mesh
        """
        if OPEN3D_AVAILABLE and isinstance(point_cloud, o3d.geometry.PointCloud):
            try:
                # Estimate normals for better reconstruction
                point_cloud.estimate_normals()
                point_cloud.orient_normals_consistent_tangent_plane(100)
                
                # Use Poisson surface reconstruction
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    point_cloud, depth=9, width=0, scale=1.1, linear_fit=False
                )
                
                # Clean the mesh based on densities
                vertices_to_remove = densities < np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(vertices_to_remove)
                
                # Convert to trimesh
                vertices = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                
                if hasattr(mesh, "vertex_colors") and np.asarray(mesh.vertex_colors).shape[0] > 0:
                    colors = np.asarray(mesh.vertex_colors)
                    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors*255)
                else:
                    return trimesh.Trimesh(vertices=vertices, faces=faces)
                
            except Exception as e:
                print(f"Error in Open3D mesh reconstruction: {e}")
                print("Falling back to basic mesh creation")
        
        # Fallback: Ball pivoting algorithm for point cloud to mesh
        # This is a simple method that may not work well for all point clouds
        points = point_cloud.vertices
        
        # Skip if too few points
        if len(points) < 100:
            return trimesh.Trimesh()
            
        # Create faces by connecting nearest neighbors
        # This is a very basic approach
        from scipy.spatial import Delaunay
        try:
            # Project points to 2D and triangulate
            # This works best for relatively flat surfaces
            # For full 3D, more sophisticated algorithms are needed
            
            # Find principal components
            mean = np.mean(points, axis=0)
            points_centered = points - mean
            cov = np.cov(points_centered.T)
            evals, evecs = np.linalg.eigh(cov)
            
            # Sort eigenvalues in descending order
            idx = np.argsort(evals)[::-1]
            evecs = evecs[:, idx]
            
            # Project onto the first two principal components
            points_2d = np.dot(points_centered, evecs[:, :2])
            
            # Generate triangulation
            tri = Delaunay(points_2d)
            
            # Create mesh
            mesh = trimesh.Trimesh(
                vertices=points,
                faces=tri.simplices,
                process=True
            )
            
            # Assign colors if available
            if hasattr(point_cloud, 'colors') and point_cloud.colors is not None:
                mesh.visual.vertex_colors = (point_cloud.colors * 255).astype(np.uint8)
                
            return mesh
            
        except Exception as e:
            print(f"Error in fallback mesh creation: {e}")
            # Return empty mesh on failure
            return trimesh.Trimesh()
    
    def _generate_rotated_views(self, image, num_views=4):
        """
        Generate images from multiple viewpoints by rotating the object.
        Using image synthesis to simulate different views.
        
        Args:
            image: Input PIL Image
            num_views: Number of different views to generate
            
        Returns:
            List of PIL Images representing different views
        """
        # Ensure text-to-image model is loaded
        self._load_models()
        
        # Original image is the front view
        views = [image]
        
        # Rotation angles (in degrees)
        angles = [90, 180, 270]  # side, back, other side
        
        # Prompts for each view
        angle_prompts = [
            "side view of the same object, detailed 3D appearance", 
            "back view of the same object, detailed 3D appearance",
            "opposite side view of the same object, detailed 3D appearance"
        ]
        
        # Get depth map for original image
        original_depth = self._estimate_depth(image)
        depth_image = PIL.Image.fromarray((original_depth * 255).astype(np.uint8))
        
        # Generate images for different viewpoints
        for i, (angle, prompt) in enumerate(zip(angles, angle_prompts)):
            # Use the model to generate a rotated view
            rotated_image = self._text_to_3d_model(
                prompt=prompt,
                image=image,
                depth_image=depth_image,
                strength=0.75,  # How much to transform the image
                num_inference_steps=30
            ).images[0]
            
            views.append(rotated_image)
            
            # Limit to requested number of views
            if len(views) >= num_views:
                break
                
        return views
        
    def generate_hunyuan3d(self, image, request_id, resolution=256, use_high_quality=True, format="glb"):
        """
        Generate a 3D model from the input image using Hunyuan3D-2 models.
        
        Args:
            image: The preprocessed input image (PIL Image)
            request_id: Unique ID for this generation request
            resolution: Output resolution
            use_high_quality: Whether to use high quality generation
            format: Output format (glb, obj)
            
        Returns:
            Path to the generated 3D model file
        """
        # Load Hunyuan3D models
        self._load_hunyuan3d_models()
        
        # Create output directory
        output_dir = Path("outputs") / request_id
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Generating 3D model with Hunyuan3D-2 for request {request_id}...")
        
        try:
            # Run the Hunyuan3D generator pipeline
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            
            # 1. Generate shape from image
            shape_mesh = self._hunyuan_shape_model(
                image=image,
                num_inference_steps=30,
                octree_resolution=380 if use_high_quality else 256,
                num_chunks=20000 if use_high_quality else 10000,
                generator=torch.manual_seed(42),
                output_type='trimesh'
            )[0]
            
            # 2. Apply texture using Hunyuan3D-Paint
            textured_mesh = self._hunyuan_texture_model(
                shape_mesh, 
                image=image
            )
            
            # 3. Save the mesh in the requested format
            output_path = output_dir / f"model.{format}"
            textured_mesh.export(output_path)
            
            # Save input image
            if hasattr(image, 'resize'):
                image.save(output_dir / "input_image.png")
            
            print(f"3D model generated with Hunyuan3D-2 and saved at {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error in Hunyuan3D-2 generation: {e}")
            
            # Fallback to traditional approach
            print("Falling back to basic method...")
            # Estimate depth map from image
            depth_map = self._estimate_depth(image)
            
            # Save depth map as visualization
            depth_vis = (depth_map * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / "depth_map.png"), depth_vis)
            
            # Create point cloud from depth map
            pcd = self._create_point_cloud_from_depth(depth_map, image, resolution=resolution)
            
            # Convert point cloud to mesh
            mesh = self._point_cloud_to_mesh(pcd, resolution=resolution)
            
            # Add texture
            if use_high_quality:
                # Create a simple texture from the image
                img_array = np.array(image.resize((1024, 1024)))
                material = texture.SimpleMaterial(image=img_array)
                
                # Apply texture
                mesh.visual = texture.TextureVisuals(material=material)
            
            # Save the mesh in the requested format
            output_path = output_dir / f"model.{format}"
            mesh.export(output_path)
            
            # Save input image
            if hasattr(image, 'resize'):
                image.save(output_dir / "input_image.png")
            
            print(f"3D model generated using fallback method and saved at {output_path}")
            return output_path
    
    def generate_with_text_prompt_hunyuan3d(self, image, prompt, request_id, resolution=256, format="glb"):
        """
        Generate a 3D model from the input image and text prompt using Hunyuan3D-2.
        
        Args:
            image: The preprocessed input image (PIL Image)
            prompt: Text description to guide the 3D generation
            request_id: Unique ID for this generation request
            resolution: Output resolution
            format: Output format (glb, obj)
            
        Returns:
            Path to the generated 3D model file
        """
        # Load Hunyuan3D models
        self._load_hunyuan3d_models()
        
        # Create output directory
        output_dir = Path("outputs") / request_id
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Generating 3D model with Hunyuan3D-2 and text prompt for request {request_id}...")
        
        try:
            # 1. Generate shape from image with text prompt
            shape_mesh = self._hunyuan_shape_model(
                image=image,
                num_inference_steps=30,
                octree_resolution=380,
                num_chunks=20000,
                text_prompt=prompt,  # Use the text prompt to guide generation
                generator=torch.manual_seed(42),
                output_type='trimesh'
            )[0]
            
            # 2. Apply texture using Hunyuan3D-Paint with text prompt
            textured_mesh = self._hunyuan_texture_model(
                shape_mesh, 
                image=image,
                text_prompt=prompt  # Use the text prompt to guide texturing
            )
            
            # 3. Save the mesh in the requested format
            output_path = output_dir / f"model.{format}"
            textured_mesh.export(output_path)
            
            # Save input image
            if hasattr(image, 'resize'):
                image.save(output_dir / "input_image.png")
            
            print(f"3D model generated with Hunyuan3D-2 and text prompt, saved at {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error in Hunyuan3D-2 generation with text prompt: {e}")
            
            # Fallback to original method
            # Ensure models are loaded
            self._load_models()
            
            print(f"Falling back to basic text-to-3D generation for request {request_id}...")
            
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
            
            print(f"3D model generated with text prompt using fallback method and saved at {output_path}")
            return output_path
    
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
        # If Hunyuan3D is available, use it for better quality
        if HUNYUAN3D_AVAILABLE:
            try:
                return self.generate_hunyuan3d(
                    image, request_id, 
                    resolution=resolution, 
                    use_high_quality=use_high_quality, 
                    format=format
                )
            except Exception as e:
                print(f"Error using Hunyuan3D-2: {e}")
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
        # If Hunyuan3D is available, use it for better quality
        if HUNYUAN3D_AVAILABLE:
            try:
                return self.generate_with_text_prompt_hunyuan3d(
                    image, prompt, request_id,
                    resolution=resolution,
                    format=format
                )
            except Exception as e:
                print(f"Error using Hunyuan3D-2 with text prompt: {e}")
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