import pytest
from app.geospatial.utils import pixel_to_geo
import rasterio
from rasterio.transform import from_origin
import numpy as np
import os

@pytest.fixture
def sample_geotiff(tmp_path):
    """Create a dummy GeoTIFF file for testing."""
    width = 100
    height = 100
    transform = from_origin(10, 40, 1, 1)

    # Create a dummy raster band
    data = np.ones((height, width), dtype=np.uint8)

    # Define the GeoTIFF metadata
    meta = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': None,
        'width': width,
        'height': height,
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': transform,
    }

    # Write the dummy GeoTIFF file
    file_path = os.path.join(tmp_path, "test.tif")
    with rasterio.open(file_path, 'w', **meta) as dst:
        dst.write(data, 1)

    return file_path

def test_pixel_to_geo(sample_geotiff):
    """Test the pixel_to_geo function."""
    # Test a coordinate in the top-left corner
    x, y = pixel_to_geo(0, 0, sample_geotiff)
    assert x == 10
    assert y == 40

    # Test a coordinate in the center
    x, y = pixel_to_geo(50, 50, sample_geotiff)
    assert x == 60
    assert y == -10

    # Test a coordinate in the bottom-right corner
    x, y = pixel_to_geo(100, 100, sample_geotiff)
    assert x == 110
    assert y == -60
