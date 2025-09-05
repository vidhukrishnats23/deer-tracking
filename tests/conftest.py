import sys
import os
import pytest
import tempfile
import shutil
from PIL import Image
import io

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from app.config import settings

@pytest.fixture(autouse=True)
def override_settings():
    """
    Override settings for tests.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        original_upload_dir = settings.upload_dir
        original_metadata_log_file = settings.metadata_log_file
        settings.upload_dir = tmpdir
        settings.metadata_log_file = f"{tmpdir}/metadata.log"
        yield
        settings.upload_dir = original_upload_dir
        settings.metadata_log_file = original_metadata_log_file

import pytest

import numpy as np
from PIL import ImageFilter

@pytest.fixture
def create_dummy_image():
    def _create_dummy_image(
        file_format: str,
        width: int,
        height: int,
        blur: bool = False,
        overexposed: bool = False,
    ) -> bytes:
        """
        Create a dummy image for testing.
        """
        if overexposed:
            # Create a white image for overexposure test
            array = np.full((height, width, 3), 255, dtype=np.uint8)
            image = Image.fromarray(array, 'RGB')
        else:
            # Create a random noise image for other tests
            array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            image = Image.fromarray(array, 'RGB')

        if blur:
            image = image.filter(ImageFilter.GaussianBlur(radius=5))

        buffer = io.BytesIO()
        image.save(buffer, format=file_format)
        return buffer.getvalue()
    return _create_dummy_image
