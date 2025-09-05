import pytest
import numpy as np
import rasterio
from app.geospatial.services import mosaic_images
import os

@pytest.fixture
def sample_geotiffs(tmp_path):
    """Create multiple sample GeoTIFF files for testing mosaicking."""
    file_paths = []
    for i in range(2):
        width, height = 10, 10
        count = 3
        dtype = 'uint8'

        data = np.random.randint(0, 255, (count, height, width), dtype=dtype)

        # Shift the origin for each image to simulate overlap
        transform = rasterio.transform.from_origin(i * 5, 10, 1, 1)

        file_path = tmp_path / f"test_{i}.tif"

        with rasterio.open(
            file_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=count,
            dtype=dtype,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(data)

        file_paths.append(str(file_path))

    return file_paths

def test_mosaic_images(sample_geotiffs, tmp_path):
    """Test the image mosaicking function."""
    output_path = tmp_path / "mosaic.tif"
    mosaic_images(sample_geotiffs, str(output_path))

    assert os.path.exists(output_path)

    with rasterio.open(output_path) as src:
        assert src.width == 15 # 10 + 5
        assert src.height == 10
        assert src.count == 3
