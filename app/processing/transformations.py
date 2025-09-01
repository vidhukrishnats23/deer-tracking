import os
from PIL import Image
from app.config import settings

def process_image(raw_image_path: str) -> str:
    """
    Apply augmentations and normalization to an image.
    """
    # Create processed directory if it doesn't exist
    os.makedirs(settings.processed_dir, exist_ok=True)

    # Open the image
    image = Image.open(raw_image_path)

    # Apply transformations
    if settings.augmentation_rotation_angle:
        image = image.rotate(settings.augmentation_rotation_angle, expand=True)

    if settings.augmentation_scale_factor:
        new_size = (int(image.width * settings.augmentation_scale_factor), int(image.height * settings.augmentation_scale_factor))
        image = image.resize(new_size)

    if settings.augmentation_flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Normalize size
    image = image.resize(settings.normalized_size)

    # Save the processed image
    base_filename = os.path.basename(raw_image_path)
    processed_image_path = os.path.join(settings.processed_dir, base_filename)
    image.save(processed_image_path)

    return processed_image_path
