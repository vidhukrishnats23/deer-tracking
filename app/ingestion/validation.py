import magic
from PIL import Image
import rasterio
import os
from app.config import settings
from app.geospatial.utils import extract_spatial_metadata

def validate_file(file):
    """
    Validate the uploaded file and extract spatial metadata if available.
    """
    error = validate_format(file)
    if error:
        return error, None

    error = validate_size(file)
    if error:
        return error, None

    if file.content_type.startswith("image/"):
        error = validate_image_corruption(file)
        if error:
            return error, None

    error = validate_resolution(file)
    if error:
        return error, None

    metadata = None
    if file.content_type == "image/tiff":
        metadata = extract_spatial_metadata(file)

    return None, metadata


def validate_image_corruption(file):
    """
    Validate if the image file is corrupted using PIL.
    """
    file.file.seek(0)
    try:
        img = Image.open(file.file)
        img.verify()
    except Exception as e:
        return f"Image file is corrupted: {e}"
    finally:
        file.file.seek(0)
    return None


def validate_format(file):
    """
    Validate the file format.
    Allowed formats are JPG, PNG, and GeoTIFF.
    """
    file.file.seek(0)
    mime_type = magic.from_buffer(file.file.read(2048), mime=True)
    file.file.seek(0)
    if mime_type not in settings.allowed_mime_types:
        return f"Invalid file format: {mime_type}. Allowed formats are {settings.allowed_mime_types}"
    return None

def validate_size(file):
    """
    Validate the file size.
    """
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > settings.max_file_size:
        return f"File size exceeds the limit of {settings.max_file_size} bytes."
    return None

def validate_resolution(file):
    """
    Validate the image resolution.
    """
    file.file.seek(0)
    try:
        if file.content_type == "image/tiff":
            with rasterio.open(file.file) as dataset:
                width = dataset.width
                height = dataset.height
        else:
            image = Image.open(file.file)
            width, height = image.size
    except Exception as e:
        return f"Could not read image resolution: {e}"
    finally:
        file.file.seek(0)

    if width > settings.max_resolution[0] or height > settings.max_resolution[1]:
        return f"Image resolution ({width}x{height}) exceeds the limit of {settings.max_resolution[0]}x{settings.max_resolution[1]} pixels."
    return None
