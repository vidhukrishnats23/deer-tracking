import pytest
import numpy as np
import rasterio
from app.processing.transformations import apply_atmospheric_correction, preprocess_winter_imagery
import os

@pytest.fixture
def sample_geotiff(tmp_path):
    """Create a sample GeoTIFF file for testing."""
    width, height = 10, 10
    count = 3
    dtype = 'uint8'

    data = np.random.randint(50, 100, (count, height, width), dtype=dtype)
    data[0, 0, 0] = 10 # Add a dark pixel for DOS

    transform = rasterio.transform.from_origin(0, 10, 1, 1)

    file_path = tmp_path / "test.tif"

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

    return str(file_path)

def test_apply_atmospheric_correction(sample_geotiff):
    """Test the atmospheric correction function."""
    with rasterio.open(sample_geotiff) as src:
        meta = src.meta.copy()
        data = src.read()

    corrected_data = apply_atmospheric_correction(data, meta)

    assert corrected_data.shape == data.shape
    assert corrected_data.dtype == data.dtype
    # The darkest pixel should be close to 0 after correction
    assert np.min(corrected_data) < 5

def test_preprocess_winter_imagery(sample_geotiff):
    """Test the winter imagery preprocessing function."""
    with rasterio.open(sample_geotiff) as src:
        meta = src.meta.copy()
        data = src.read()

    enhanced_data = preprocess_winter_imagery(data, meta)

    assert enhanced_data.shape == data.shape
    assert enhanced_data.dtype == data.dtype
    # Check that the contrast has been enhanced (variance should be higher)
    assert np.var(enhanced_data) > np.var(data)
