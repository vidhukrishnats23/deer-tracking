import magic
from PIL import Image
import rasterio
import os
import cv2
import numpy as np
from app.config import settings
from app.geospatial.utils import extract_detailed_metadata

def validate_file(file):
    """
    Validate the uploaded file and extract detailed metadata if available.
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

    metadata = extract_detailed_metadata(file)

    if file.content_type.startswith("image/"):
        error = validate_exposure(file)
        if error:
            return error, metadata

        error = validate_blurriness(file)
        if error:
            return error, metadata

    if metadata.get("exif"):
        error = validate_gsd(metadata)
        if error:
            return error, metadata

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


def validate_blurriness(file):
    """
    Validate if the image is blurry using Laplacian variance.
    """
    file.file.seek(0)
    try:
        image_stream = file.file.read()
        image_array = np.frombuffer(image_stream, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            return "Could not decode image for blurriness check."
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < settings.blur_threshold:
            return f"Image is likely blurry. Laplacian variance: {laplacian_var:.2f} (threshold: {settings.blur_threshold})"
    except Exception as e:
        return f"Could not perform blurriness check: {e}"
    finally:
        file.file.seek(0)
    return None


def validate_exposure(file):
    """
    Validate image exposure by checking for under or over-exposure.
    """
    file.file.seek(0)
    try:
        image_stream = file.file.read()
        image_array = np.frombuffer(image_stream, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return "Could not decode image for exposure check."

        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()

        if hist_norm[0] > settings.underexposure_threshold:
            return f"Image is likely underexposed. Black pixel percentage: {hist_norm[0]:.2%}"
        if hist_norm[-1] > settings.overexposure_threshold:
            return f"Image is likely overexposed. White pixel percentage: {hist_norm[-1]:.2%}"

    except Exception as e:
        return f"Could not perform exposure check: {e}"
    finally:
        file.file.seek(0)
    return None


def validate_gsd(metadata):
    """
    Validate the Ground Sampling Distance (GSD).
    Requires focal length, sensor width, and altitude from EXIF data.
    """
    try:
        exif_data = metadata.get("exif", {})
        focal_length_str = exif_data.get("EXIF FocalLength")
        altitude_str = exif_data.get("GPS GPSAltitude")

        if not focal_length_str or not altitude_str:
            return None # Not enough data to calculate GSD

        focal_length = float(str(focal_length_str).split('/')[0]) / float(str(focal_length_str).split('/')[1]) if '/' in str(focal_length_str) else float(str(focal_length_str))
        altitude = float(str(altitude_str).split('/')[0]) / float(str(altitude_str).split('/')[1]) if '/' in str(altitude_str) else float(str(altitude_str))

        # Assume sensor width is a known parameter, for now, get from settings
        sensor_width_mm = settings.sensor_width_mm
        image_width_px = metadata.get("spatial", {}).get("width")

        if not image_width_px:
            return "Image width not available for GSD calculation."

        gsd = (altitude * sensor_width_mm) / (focal_length * image_width_px) # in m/px

        if gsd > settings.max_gsd:
            return f"GSD ({gsd:.2f} m/px) exceeds the limit of {settings.max_gsd} m/px."

    except (ValueError, TypeError, ZeroDivisionError) as e:
        return f"Could not calculate GSD from EXIF data: {e}"
    return None
