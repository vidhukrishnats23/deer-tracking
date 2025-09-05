from app.geospatial.services import mosaic_images as mosaic_images_geospatial
from app.config import settings
import os
from typing import List

def create_mosaic(image_paths: List[str], output_filename: str):
    """
    Create a mosaic from a list of images.
    """
    # Ensure the image paths are absolute, assuming they are relative to the upload_dir
    absolute_image_paths = [os.path.join(settings.upload_dir, path) for path in image_paths]

    # Ensure the output path is absolute, assuming it's relative to the processed_dir
    output_path = os.path.join(settings.processed_dir, output_filename)

    mosaic_images_geospatial(absolute_image_paths, output_path)

    return output_path
