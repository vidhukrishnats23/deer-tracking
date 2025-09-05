import os
from fastapi import UploadFile
from app.config import settings
from . import validation
from app.logger import logger

def save_annotation(file: UploadFile):
    """
    Save the uploaded annotation file.
    """
    try:
        file.file.seek(0)

        # Create labels directory if it doesn't exist
        os.makedirs(settings.labels_dir, exist_ok=True)

        file_path = os.path.join(settings.labels_dir, file.filename)

        # Save the file
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        logger.info(f"Saved annotation file {file.filename} to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving annotation file {file.filename}: {e}")
        raise e

def validate_annotation(file_path: str):
    """
    Validate the annotation file.
    """
    logger.info(f"Validating annotation file: {file_path}")
    error = validation.validate_yolo_annotation(file_path)
    if error:
        logger.warning(f"Validation failed for {file_path}: {error}")
    else:
        logger.info(f"Validation successful for {file_path}")
    return error
