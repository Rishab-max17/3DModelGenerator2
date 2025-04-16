import shutil
from pathlib import Path
from fastapi import UploadFile

async def save_upload(file: UploadFile, request_id: str) -> Path:
    """
    Save an uploaded file to the uploads directory.
    
    Args:
        file: The uploaded file
        request_id: Unique ID for this request
        
    Returns:
        Path to the saved file
    """
    # Create directory for this request
    upload_dir = Path("uploads") / request_id
    upload_dir.mkdir(exist_ok=True, parents=True)
    
    # Save file
    file_extension = file.filename.split('.')[-1]
    file_path = upload_dir / f"input.{file_extension}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return file_path

def cleanup_files(request_id: str, keep_outputs=True):
    """
    Clean up temporary files after request processing.
    
    Args:
        request_id: Unique ID for the request
        keep_outputs: Whether to keep output files
    """
    # Remove upload directory
    upload_dir = Path("uploads") / request_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    
    # Optionally remove output directory
    if not keep_outputs:
        output_dir = Path("outputs") / request_id
        if output_dir.exists():
            shutil.rmtree(output_dir) 