import os
from fastapi import UploadFile
from app.config import settings
from . import validation

def save_annotation(file: UploadFile):
    """
    Save the uploaded annotation file.
    """
    file.file.seek(0)

    # Create labels directory if it doesn't exist
    os.makedirs(settings.labels_dir, exist_ok=True)

    file_path = os.path.join(settings.labels_dir, file.filename)

    # Save the file
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    return file_path

def validate_annotation(file_path: str):
    """
    Validate the annotation file.
    """
    return validation.validate_yolo_annotation(file_path)
