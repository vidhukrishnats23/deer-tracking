import os
import rasterio
import numpy as np
from PIL import Image
import cv2
from app.config import settings
from app.logger import logger
from typing import List, Optional

def apply_atmospheric_correction(data, meta):
    """
    Apply a simple Dark Object Subtraction (DOS) atmospheric correction.
    """
    dark_object = np.percentile(data, 1, axis=(1, 2))
    corrected_data = data - dark_object[:, np.newaxis, np.newaxis]
    corrected_data[corrected_data < 0] = 0
    logger.info("Applied atmospheric correction.")
    return corrected_data.astype(meta['dtype'])

def apply_radiometric_calibration(data, meta):
    """
    Placeholder for radiometric calibration.
    This would convert DN to radiance or reflectance.
    Requires sensor-specific information.
    """
    logger.info("Radiometric calibration placeholder.")
    return data

def preprocess_winter_imagery(data, meta):
    """
    Preprocess winter imagery, e.g., by applying contrast enhancement.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        # Ensure data is in a format that CLAHE can handle (e.g., uint8 or uint16)
        if data.dtype == 'uint8' or data.dtype == 'uint16':
            enhanced_data[i] = clahe.apply(data[i])
        else:
            # If not, normalize, apply, and scale back
            band = data[i]
            band_norm = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            enhanced_band = clahe.apply(band_norm)
            enhanced_data[i] = cv2.normalize(enhanced_band, None, np.min(band), np.max(band), cv2.NORM_MINMAX, dtype=data.dtype)
    logger.info("Applied winter imagery preprocessing.")
    return enhanced_data

def process_image(
    raw_image_path: str,
    season: Optional[str] = None,
    processing_pipeline: Optional[List[str]] = None
) -> str:
    """
    Apply a flexible pipeline of processing steps to an image.
    """
    os.makedirs(settings.processed_dir, exist_ok=True)
    base_filename = os.path.basename(raw_image_path)
    processed_image_path = os.path.join(settings.processed_dir, base_filename)

    if processing_pipeline is None:
        processing_pipeline = []

    try:
        with rasterio.open(raw_image_path) as src:
            meta = src.meta.copy()
            data = src.read()

            if not src.crs:
                raise rasterio.errors.RasterioIOError("Not a georeferenced raster.")

            if season == 'winter':
                data = preprocess_winter_imagery(data, meta)

            if 'atmospheric_correction' in processing_pipeline:
                data = apply_atmospheric_correction(data, meta)

            if 'radiometric_calibration' in processing_pipeline:
                data = apply_radiometric_calibration(data, meta)

            # --- Augmentation and Normalization for GeoTIFFs ---
            height, width = data.shape[1], data.shape[2]
            normalized_height, normalized_width = settings.normalized_size

            resampled_data = np.zeros((src.count, normalized_height, normalized_width), dtype=data.dtype)
            for i in range(src.count):
                img = Image.fromarray(data[i])
                resampled_img = img.resize((normalized_width, normalized_height), Image.Resampling.BILINEAR)
                resampled_data[i] = np.array(resampled_img)

            transform = src.transform * src.transform.scale(
                (width / normalized_width),
                (height / normalized_height)
            )

            meta.update({
                "height": normalized_height,
                "width": normalized_width,
                "transform": transform,
                "driver": "GTiff"
            })

            with rasterio.open(processed_image_path, 'w', **meta) as dst:
                dst.write(resampled_data)

            logger.info(f"Processed {raw_image_path} as GeoTIFF with pipeline: {processing_pipeline}")

    except (rasterio.errors.RasterioIOError, AttributeError):
        logger.warning(f"Could not open {raw_image_path} with rasterio. Falling back to PIL.")
        image = Image.open(raw_image_path)

        if settings.augmentation_rotation_angle:
            image = image.rotate(settings.augmentation_rotation_angle, expand=True)
        if settings.augmentation_scale_factor:
            new_size = (int(image.width * settings.augmentation_scale_factor), int(image.height * settings.augmentation_scale_factor))
            image = image.resize(new_size)
        if settings.augmentation_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image = image.resize(settings.normalized_size)
        image.save(processed_image_path)
        logger.info(f"Processed {raw_image_path} with PIL.")

    return processed_image_path
