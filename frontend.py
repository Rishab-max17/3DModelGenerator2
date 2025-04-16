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

# Optional text prompt
use_text_prompt = st.sidebar.checkbox("Use Text Prompt", value=False)
text_prompt = st.sidebar.text_input("Text Prompt (optional)", value="A detailed 3D sculpture", 
                                   disabled=not use_text_prompt)

# Multi-view option (only show for Default model)
if model_type == "Default":
    multi_view = st.sidebar.checkbox("Generate from Multiple Angles", value=True)
else:
    # Hunyuan3D-2 always uses multi-view generation
    multi_view = True

# API URL (change if needed)
API_URL = "http://localhost:8000"

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
    
    # Create a temporary file for the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Show progress
    with st.spinner(f"Generating 3D model using {model_type}... This may take a while."):
        progress_bar = st.progress(0)
        
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
                    'use_hunyuan': True
                }
                
                # Make the request
                for i in range(1, 101):
                    time.sleep(0.05)  # Simulate progress
                    progress_bar.progress(i)
                    
                    if i == 30:
                        response = requests.post(endpoint, files=files, data=data)
                        if response.status_code == 200:
                            result = response.json()
                            request_id = result.get("request_id")
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                            break
            else:
                # Use standard endpoint with query parameters instead of form data
                endpoint = f"{API_URL}/generate-simple"
                files = {'file': open(tmp_file_path, 'rb')}
                
                # Send parameters as query parameters
                params = {
                    'resolution': resolution,
                    'use_high_quality': high_quality,
                    'format': model_format,
                    'multi_view': multi_view,
                    'use_hunyuan': True
                }
                
                # Make the request using query parameters
                for i in range(1, 101):
                    time.sleep(0.05)  # Simulate progress
                    progress_bar.progress(i)
                    
                    if i == 30:
                        response = requests.post(endpoint, files=files, params=params)
                        if response.status_code == 200:
                            result = response.json()
                            request_id = result.get("request_id")
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                            break
                            
        except Exception as e:
            st.error(f"Failed to connect to API: {str(e)}")
            
        finally:
            # Clean up the temporary file
            os.unlink(tmp_file_path)
    
    # After successful generation
    if 'request_id' in locals():
        with col2:
            st.subheader("Generated Output")
            
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
            
            # Download link for the 3D model
            model_url = f"{API_URL}/models/{request_id}/download"
            st.markdown(f"**[Download 3D Model]({model_url})**")
            
            if model_type == "Hunyuan3D-2":
                st.success("3D model generated with Hunyuan3D-2 for high-quality results from all angles!")
            
            # Add viewer instructions
            st.info("To view the 3D model, download it and open with a 3D viewer like Blender or upload to an online viewer.")

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
    6. Wait for the model to be generated
    7. Download the 3D model file
    8. View the model using a 3D viewer like Blender
    """) 