import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def preprocess_image(image_path):
    """
    Preprocess the input image for the model.
    
    Args:
        image_path: Path to the uploaded image
        
    Returns:
        Preprocessed image ready for the model
    """
    # Load image
    img = cv2.imread(str(image_path))
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to standard size (e.g., 512x512)
    img = cv2.resize(img, (512, 512))
    
    # Convert to PIL Image which is required by the model
    pil_img = Image.fromarray(img)
    
    return pil_img 