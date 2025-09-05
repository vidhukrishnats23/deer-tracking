import uuid
import os
import datetime
import json
from app.config import settings
from fastapi import UploadFile
from app.processing.transformations import process_image
from app.geospatial.services import reproject_image, orthorectify_image
from app.logger import logger
from typing import Optional

def log_metadata(metadata: dict):
    """
    Log metadata about the ingested image to a file.
    """
    with open(settings.metadata_log_file, "a") as f:
        f.write(json.dumps(metadata) + "\n")

from typing import List

async def save_file(
    file: UploadFile,
    detailed_metadata: dict = None,
    season: Optional[str] = None,
    processing_pipeline: Optional[List[str]] = None
):
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
            buffer.write(await file.read())

        # Atomically move the file to its final destination
        os.rename(temp_file_path, final_file_path)
        file_path = final_file_path
        logger.info(f"Saved file {file.filename} to {file_path}")

        # Reproject the image if it has spatial metadata and a different CRS
        if detailed_metadata and detailed_metadata.get('spatial') and detailed_metadata['spatial'].get('crs') and detailed_metadata['spatial']['crs'] != settings.TARGET_CRS:
            reprojected_path = os.path.join(settings.processed_dir, f"reprojected_{unique_filename}")
            reproject_image(file_path, reprojected_path)
            logger.info(f"Reprojected image for {file.filename} and saved to {reprojected_path}")
            file_path = reprojected_path

        # Orthorectify the image if enabled
        if settings.APPLY_ORTHO_ON_INGEST:
            ortho_path = os.path.join(settings.processed_dir, f"ortho_{unique_filename}")
            orthorectify_image(file_path, ortho_path)
            logger.info(f"Orthorectified image for {file.filename} and saved to {ortho_path}")
            file_path = ortho_path

        # Process the image
        processed_image_path = process_image(
            file_path,
            season=season,
            processing_pipeline=processing_pipeline
        )
        logger.info(f"Processed image for {file.filename} and saved to {processed_image_path}")

        # Get file size
        file_size = os.path.getsize(file_path)

        # Get image resolution
        from . import validation
        file.file.seek(0) # Reset file pointer before reading again
        resolution_error = validation.validate_resolution(file)
        if resolution_error:
             logger.warning(f"Could not validate resolution for {file.filename}: {resolution_error}")
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
            "detailed_metadata": detailed_metadata
        }
        if season:
            metadata['season'] = season

        log_metadata(metadata)

        logger.info(f"Generated metadata for {file.filename}: {metadata}")
        return metadata
    except Exception as e:
        logger.error(f"Error saving file {file.filename}: {e}")
        raise e
