import os
import rasterio
import numpy as np
from PIL import Image
from app.config import settings
from app.logger import logger

def process_image(raw_image_path: str) -> str:
    """
    Apply augmentations and normalization to an image.
    If the image is a GeoTIFF, it attempts to preserve georeferencing.
    Otherwise, it falls back to standard PIL processing.
    """
    os.makedirs(settings.processed_dir, exist_ok=True)
    base_filename = os.path.basename(raw_image_path)
    processed_image_path = os.path.join(settings.processed_dir, base_filename)

    try:
        # Try processing as a GeoTIFF to preserve metadata
        with rasterio.open(raw_image_path) as src:
            meta = src.meta.copy()

            # If no CRS, it's not a georeferenced image, fall back to PIL
            if not src.crs:
                raise rasterio.errors.RasterioIOError("Not a georeferenced raster.")

            data = src.read()
            height, width = data.shape[1], data.shape[2]

            normalized_height, normalized_width = settings.normalized_size

            # Resample data to target size using PIL on each band
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

            logger.info(f"Processed {raw_image_path} as GeoTIFF.")

    except (rasterio.errors.RasterioIOError, AttributeError):
        # Fallback to standard PIL processing for non-GeoTIFFs or non-georeferenced files
        logger.warning(f"Could not open {raw_image_path} with rasterio or not georeferenced. Falling back to PIL.")
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
