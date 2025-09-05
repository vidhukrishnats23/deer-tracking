import magic
from PIL import Image
from fastapi import UploadFile, HTTPException
from app.config import settings

def validate_image(file: UploadFile):
    """
    Validate that the uploaded file is a valid image.
    """
    # Validate format
    file.file.seek(0)
    mime_type = magic.from_buffer(file.file.read(2048), mime=True)
    file.file.seek(0)
    if mime_type not in settings.allowed_mime_types:
        raise HTTPException(status_code=400, detail=f"Invalid file format: {mime_type}. Allowed formats are {settings.allowed_mime_types}")

    # Validate image corruption
    try:
        img = Image.open(file.file)
        img.verify()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image file is corrupted: {e}")
    finally:
        file.file.seek(0)
