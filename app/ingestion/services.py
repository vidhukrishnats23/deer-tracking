import uuid
import os
import datetime
import json
from app.config import settings
from fastapi import UploadFile
from app.processing.transformations import process_image
from app.geospatial.services import reproject_image
from app.logger import logger

def save_file(file: UploadFile, spatial_metadata: dict = None):
    """
    Save the uploaded file with a unique filename and process it.
    """
    try:
        file.file.seek(0)

        # Create upload directory if it doesn't exist
        os.makedirs(settings.upload_dir, exist_ok=True)

        # Generate a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"

        # Create a temporary file path
        temp_file_path = os.path.join(settings.upload_dir, f"temp_{unique_filename}")
        final_file_path = os.path.join(settings.upload_dir, unique_filename)

        # Save the file to a temporary location
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Atomically move the file to its final destination
        os.rename(temp_file_path, final_file_path)
        file_path = final_file_path
        logger.info(f"Saved file {file.filename} to {file_path}")

        # Reproject the image if it has spatial metadata and a different CRS
        if spatial_metadata and spatial_metadata.get('crs') and spatial_metadata['crs'] != settings.TARGET_CRS:
            reprojected_path = os.path.join(settings.processed_dir, f"reprojected_{unique_filename}")
            reproject_image(file_path, reprojected_path)
            logger.info(f"Reprojected image for {file.filename} and saved to {reprojected_path}")
            file_path = reprojected_path

        # Process the image
        processed_image_path = process_image(file_path)
        logger.info(f"Processed image for {file.filename} and saved to {processed_image_path}")

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
            "processed_file_path": processed_image_path,
            "file_size": file_size,
            "resolution": f"{width}x{height}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "augmentations_applied": {
                "rotation_angle": settings.augmentation_rotation_angle,
                "scale_factor": settings.augmentation_scale_factor,
                "flipped": settings.augmentation_flip,
                "normalized_size": settings.normalized_size,
            },
            "spatial_metadata": spatial_metadata
        }
        logger.info(f"Generated metadata for {file.filename}: {metadata}")
        return metadata
    except Exception as e:
        logger.error(f"Error saving file {file.filename}: {e}")
        raise e
