import streamlit as st
import requests
import json
import os
from PIL import Image
import tempfile
import time
import base64

# Page config
st.set_page_config(page_title="2D to 3D Model Generator", layout="wide")

# Add 3D model viewer component
def render_3d_viewer(model_url):
    """Render a 3D model viewer for GLB or OBJ files"""
    model_viewer_html = f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer@1.12.0/dist/model-viewer.min.js"></script>
    <style>
    model-viewer {{
      width: 100%;
      height: 400px;
      background-color: #f5f5f5;
      --poster-color: transparent;
    }}
    </style>
    <model-viewer
      src="{model_url}"
      alt="3D Model"
      auto-rotate
      camera-controls
      shadow-intensity="1"
      exposure="0.7"
      ar
      ar-modes="webxr scene-viewer quick-look"
    ></model-viewer>
    """
    st.components.v1.html(model_viewer_html, height=450)

# App title and description
st.title("2D to 3D Model Generator")
st.markdown("Upload a 2D image and convert it to a 3D model using deep learning.")

# Sidebar options
st.sidebar.header("Model Options")

# Model selection
model_type = st.sidebar.radio(
    "3D Generation Model", 
    ["Hunyuan3D-2 (recommended)", "Default"],
    index=0
)

resolution = st.sidebar.slider("Resolution", min_value=128, max_value=512, value=256, step=64)
high_quality = st.sidebar.checkbox("High Quality", value=True)
model_format = st.sidebar.selectbox("Output Format", options=["glb", "obj"], index=0)

# Enhanced Model Options
with st.sidebar.expander("Enhanced Model Options", expanded=False):
    texture_resolution = st.slider("Texture Resolution", min_value=1024, max_value=4096, value=2048, step=512, 
                                help="Higher values create more detailed textures but take longer")
    remove_background = st.checkbox("Remove Background", value=True, 
                                  help="Automatically remove background from image")
    geometry_detail_level = st.selectbox("Geometry Detail", 
                                      options=["low", "medium", "high"], 
                                      index=2,
                                      help="Level of geometric detail in the model")
    texture_quality = st.selectbox("Texture Quality", 
                                options=["basic", "medium", "high"], 
                                index=2,
                                help="Quality of textures applied to the model")

# Advanced options collapsible section
with st.sidebar.expander("Advanced Options"):
    retry_count = st.number_input("Max Retries", min_value=1, max_value=5, value=2, 
                                 help="Number of times to retry generation if it fails")
    fallback_enabled = st.checkbox("Enable Fallback Generation", value=True,
                                  help="If model generation fails, fall back to simpler methods")
    model_version = st.sidebar.selectbox(
        "Model Version", 
        ["b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb",
         "3af0e5919e8c1d3fc52ef9e67584b6364a287e67d36ee95ef0f3b3425a484879"],
        index=0,
        help="Specific model version to use (advanced)"
    )

# Optional text prompt
use_text_prompt = st.sidebar.checkbox("Use Text Prompt", value=False)
text_prompt = st.sidebar.text_input(
    "Text Prompt (optional)", 
    value="A detailed 3D model with clear shape and texture", 
    disabled=not use_text_prompt,
    help="Describe what you want the model to look like"
)

# Multi-view option (only show for Default model)
if model_type == "Default":
    multi_view = st.sidebar.checkbox("Generate from Multiple Angles", value=True)
else:
    # Hunyuan3D-2 always uses multi-view generation
    multi_view = True

# API URL (change if needed)
API_URL = "http://localhost:8003"

# Function to create download link for files
def get_download_link(file_path, link_text):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    return href

# Main content
with st.form("upload_form"):
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    submit_button = st.form_submit_button("Generate 3D Model")

# Process the uploaded file when submitted
if uploaded_file is not None and submit_button:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
        # Show image recommendations
        with st.expander("Image Tips"):
            st.markdown("""
            For best results:
            - Use images with clear objects
            - Ensure the subject is well-lit
            - Avoid complex backgrounds
            - Provide images with good contrast
            """)
    
    # Create a temporary file for the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Show progress
    with st.spinner(f"Generating 3D model using {model_type}... This may take 1-3 minutes."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Create the request parameters
            if use_text_prompt:
                # Use text prompt endpoint
                endpoint = f"{API_URL}/generate-with-prompt"
                files = {'file': open(tmp_file_path, 'rb')}
                
                # For text prompt, send parameters directly without using JSON string
                data = {
                    'prompt': text_prompt,
                    'resolution': str(resolution),
                    'format': model_format,
                    'use_hunyuan': True,
                    'retry_count': retry_count,
                    'model_version': model_version,
                    'texture_resolution': texture_resolution,
                    'remove_background': remove_background,
                    'geometry_detail_level': geometry_detail_level,
                    'texture_quality': texture_quality
                }
                
                # Update status
                status_text.text("Initializing model generation...")
                
                # Make the request
                for i in range(1, 101):
                    progress_percentage = min(i, 95)  # Cap at 95% until we get the result
                    progress_bar.progress(progress_percentage)
                    
                    if i < 20:
                        status_text.text("Preparing image for processing...")
                    elif i < 40:
                        status_text.text("Generating depth information...")
                    elif i < 60:
                        status_text.text("Creating 3D geometry...")
                    elif i < 80:
                        status_text.text("Applying textures...")
                    elif i < 95:
                        status_text.text("Finalizing model...")
                    
                    # Actual API call at 30% progress
                    if i == 30:
                        status_text.text("Sending to Hunyuan3D API...")
                        response = requests.post(endpoint, files=files, data=data)
                        if response.status_code == 200:
                            result = response.json()
                            request_id = result.get("request_id")
                            status_text.text("Model generation successful! Preparing for display...")
                        else:
                            error_msg = f"Error: {response.status_code}"
                            try:
                                error_details = response.json().get("detail", "Unknown error")
                                error_msg += f" - {error_details}"
                            except:
                                error_msg += f" - {response.text}"
                            
                            st.error(error_msg)
                            status_text.text("Generation failed. Please try again.")
                            break
                    
                    time.sleep(0.05)  # Simulate progress for UI
            else:
                # Use standard endpoint with query parameters
                endpoint = f"{API_URL}/generate-simple"
                files = {'file': open(tmp_file_path, 'rb')}
                
                # Send parameters as query parameters
                params = {
                    'resolution': resolution,
                    'use_high_quality': high_quality,
                    'format': model_format,
                    'multi_view': multi_view,
                    'use_hunyuan': True,
                    'retry_count': retry_count,
                    'model_version': model_version,
                    'texture_resolution': texture_resolution,
                    'remove_background': remove_background,
                    'geometry_detail_level': geometry_detail_level,
                    'texture_quality': texture_quality
                }
                
                # Update status
                status_text.text("Initializing model generation...")
                
                # Make the request using query parameters
                for i in range(1, 101):
                    progress_percentage = min(i, 95)  # Cap at 95% until we get the result
                    progress_bar.progress(progress_percentage)
                    
                    if i < 20:
                        status_text.text("Preparing image for processing...")
                    elif i < 40:
                        status_text.text("Generating depth information...")
                    elif i < 60:
                        status_text.text("Creating 3D geometry...")
                    elif i < 80:
                        status_text.text("Applying textures...")
                    elif i < 95:
                        status_text.text("Finalizing model...")
                    
                    # Actual API call at 30% progress
                    if i == 30:
                        status_text.text("Sending to Hunyuan3D API...")
                        response = requests.post(endpoint, files=files, params=params)
                        if response.status_code == 200:
                            result = response.json()
                            request_id = result.get("request_id")
                            status_text.text("Model generation successful! Preparing for display...")
                        else:
                            error_msg = f"Error: {response.status_code}"
                            try:
                                error_details = response.json().get("detail", "Unknown error")
                                error_msg += f" - {error_details}"
                            except:
                                error_msg += f" - {response.text}"
                            
                            st.error(error_msg)
                            status_text.text("Generation failed. Please try again.")
                            break
                    
                    time.sleep(0.05)  # Simulate progress for UI
                    
            # Final progress update
            if 'request_id' in locals():
                progress_bar.progress(100)
                status_text.text("Model generation complete!")
                    
        except Exception as e:
            st.error(f"Failed to generate 3D model: {str(e)}")
            status_text.text("An error occurred during generation. Please try again.")
            
        finally:
            # Clean up the temporary file
            os.unlink(tmp_file_path)
    
    # After successful generation
    if 'request_id' in locals():
        with col2:
            st.subheader("Generated Output")
            
            # Get model info to verify it's a proper 3D model
            model_info_url = f"{API_URL}/models/{request_id}/info"
            try:
                model_info_response = requests.get(model_info_url)
                if model_info_response.status_code == 200:
                    model_info = model_info_response.json()
                    
                    # Check if model has proper 3D metrics
                    vertices_count = model_info.get("vertices_count", 0)
                    faces_count = model_info.get("faces_count", 0)
                    
                    if vertices_count < 100 or faces_count < 100:
                        st.warning("⚠️ The generated model is simplified due to API limitations. We've created a basic 3D shape as a fallback.")
                        st.info("To get better quality models, please check if your Replicate API token is valid and has sufficient credits.")
                    else:
                        st.success(f"✅ Successfully generated a 3D model with {vertices_count} vertices and {faces_count} faces.")
                        
                    # Show extra info about the model dimensions if available
                    if "bounding_box" in model_info:
                        st.info(f"Model dimensions: {model_info['bounding_box']}")
            except Exception as e:
                st.error(f"Could not get model info: {e}")
                st.info("We'll still try to display a model preview below.")
            
            # Get and display depth map or preview image
            try:
                if model_type == "Default":
                    preview_url = f"{API_URL}/models/{request_id}/depth_map.png"
                    preview_caption = "Depth Map"
                else:
                    preview_url = f"{API_URL}/models/{request_id}/input_image.png"
                    preview_caption = "Input Image"
                    
                preview_response = requests.get(preview_url)
                if preview_response.status_code == 200:
                    st.image(preview_response.content, caption=preview_caption, use_container_width=True)
            except:
                st.write("Preview image not available")
            
            # 3D model viewer
            model_url = f"{API_URL}/models/{request_id}/download"
            st.subheader("3D Model Preview")
            st.markdown("Preview your 3D model below (requires WebGL support):")
            
            # Add troubleshooting info for API issues
            with st.expander("Troubleshooting API Issues"):
                st.markdown("""
                If you're seeing a simplified model or cube instead of your expected 3D model:
                
                1. **API Token**: Make sure your Replicate API token is valid and has sufficient credits
                2. **Image Quality**: Try using a different image with a clear subject and good lighting
                3. **Advanced Options**: Try a different model version or increase resolution
                4. **Prompting**: If using text prompts, be specific about the 3D shape you expect
                
                You can get a new API token from [Replicate's website](https://replicate.com/account/api-tokens).
                """)
            
            # Show the 3D model viewer
            render_3d_viewer(model_url)
            
            # Download link for the 3D model
            st.markdown(f"**[Download 3D Model]({model_url})**")
            
            if model_type == "Hunyuan3D-2":
                if vertices_count < 100 or faces_count < 100:
                    st.info("A fallback simple model was generated since the Hunyuan3D-2 API request failed.")
                else:
                    st.success("3D model generated with Hunyuan3D-2 for high-quality results from all angles!")

# Add model information
st.sidebar.markdown("---")
if model_type == "Hunyuan3D-2":
    st.sidebar.markdown("""
    **About Hunyuan3D-2**
    
    Hunyuan3D-2 is Tencent's advanced 3D model generation system that creates
    high-quality textured 3D models from a single image. It features:
    
    - Better geometry generation
    - High-resolution texturing
    - Full 360° model creation
    - More accurate results
    """)

# Add instructions
with st.expander("How to use"):
    st.markdown("""
    1. Upload a 2D image
    2. Choose between Hunyuan3D-2 (recommended) or Default model
    3. Adjust resolution and quality settings
    4. Optionally add a text prompt to guide the 3D generation
    5. Click "Generate 3D Model"
    6. Wait for the model to be generated (this can take several minutes)
    7. Preview the 3D model directly in your browser
    8. Download the 3D model file for use in your projects
    
    **Note:** The 3D model preview requires WebGL support in your browser.
    """) 