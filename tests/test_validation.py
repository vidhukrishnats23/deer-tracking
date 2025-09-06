import pytest
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from app.geospatial.services import calculate_morans_i

@pytest.fixture
def gdf_clustered():
    """GeoDataFrame with a clustered pattern."""
    points = [
        Point(0, 0), Point(1, 0), Point(0, 1),
        Point(10, 10), Point(11, 10), Point(10, 11)
    ]
    values = [1, 1.1, 0.9, 10, 10.1, 9.9]
    return gpd.GeoDataFrame({'value': values}, geometry=points)

@pytest.fixture
def gdf_dispersed():
    """GeoDataFrame with a dispersed pattern."""
    points = [Point(0, 0), Point(10, 10), Point(0, 10), Point(10, 0)]
    values = [10, 10, 1, 1]
    return gpd.GeoDataFrame({'value': values}, geometry=points)

@pytest.fixture
def gdf_random():
    """GeoDataFrame with a random pattern."""
    np.random.seed(42)
    points = [Point(x, y) for x, y in np.random.rand(100, 2) * 10]
    values = np.random.rand(100)
    return gpd.GeoDataFrame({'value': values}, geometry=points)

def test_calculate_morans_i_clustered(gdf_clustered):
    """Test Moran's I for a clustered pattern. Expect positive autocorrelation."""
    morans_i, p_value = calculate_morans_i(gdf_clustered, 'value')
    assert morans_i is not None
    assert morans_i > 0

def test_calculate_morans_i_dispersed(gdf_dispersed):
    """Test Moran's I for a dispersed pattern. Expect negative autocorrelation."""
    morans_i, p_value = calculate_morans_i(gdf_dispersed, 'value')
    assert morans_i is not None
    assert morans_i < 0

def test_calculate_morans_i_random(gdf_random):
    """Test Moran's I for a random pattern. Expect no significant autocorrelation."""
    morans_i, p_value = calculate_morans_i(gdf_random, 'value')
    assert morans_i is not None
    # For a random pattern, Moran's I should be close to -1/(n-1)
    expected_morans_i = -1 / (len(gdf_random) - 1)
    assert np.isclose(morans_i, expected_morans_i, atol=0.1)
