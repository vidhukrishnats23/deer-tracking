import uuid
import os
import datetime
import json
from app.config import settings
from fastapi import UploadFile

def save_file(file: UploadFile):
    """
    Save the uploaded file with a unique filename.
    """
    file.file.seek(0)

    # Generate a unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(settings.upload_dir, unique_filename)

    # Save the file
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    # Get file size
    file_size = os.path.getsize(file_path)

    # Get image resolution
    from . import validation
    resolution = validation.validate_resolution(file)
    if isinstance(resolution, str): # An error occurred
        width, height = -1, -1
    else:
        file.file.seek(0)
        if file.content_type == "image/tiff":
            import rasterio
            with rasterio.open(file.file) as dataset:
                width = dataset.width
                height = dataset.height
        else:
            from PIL import Image
            image = Image.open(file.file)
            width, height = image.size

    metadata = {
        "original_filename": file.filename,
        "unique_filename": unique_filename,
        "file_path": file_path,
        "file_size": file_size,
        "resolution": f"{width}x{height}",
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    return metadata

def log_metadata(metadata: dict):
    """
    Log the metadata of the uploaded file.
    """
    with open(settings.metadata_log_file, "a") as f:
        f.write(json.dumps(metadata) + "\n")
