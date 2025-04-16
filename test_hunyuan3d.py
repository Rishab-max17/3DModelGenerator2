import argparse
from pathlib import Path
from PIL import Image
import os
from model.generator import ModelGenerator

def main():
    parser = argparse.ArgumentParser(description='Test 3D model generation with Hunyuan3D-2')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='test_output', help='Output directory')
    parser.add_argument('--text_prompt', type=str, help='Optional text prompt to guide generation')
    parser.add_argument('--high_quality', action='store_true', help='Use high quality settings')
    parser.add_argument('--format', type=str, default='glb', choices=['glb', 'obj'], help='Output format')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file {image_path} not found")
        return
    
    image = Image.open(image_path)
    
    # Initialize model generator
    model_generator = ModelGenerator()
    
    # Generate unique request ID
    request_id = image_path.stem
    
    print(f"Generating 3D model from image: {image_path}")
    print(f"Output directory: {output_dir}")
    
    if args.text_prompt:
        print(f"Using text prompt: {args.text_prompt}")
        output_path = model_generator.generate_with_text_prompt_hunyuan3d(
            image,
            args.text_prompt,
            request_id,
            resolution=512 if args.high_quality else 256,
            format=args.format
        )
    else:
        output_path = model_generator.generate_hunyuan3d(
            image,
            request_id,
            resolution=512 if args.high_quality else 256,
            use_high_quality=args.high_quality,
            format=args.format
        )
    
    print(f"3D model generated and saved at: {output_path}")
    print(f"To view the model, use a 3D viewer like Blender or an online viewer.")

if __name__ == "__main__":
    main() 