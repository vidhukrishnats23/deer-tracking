import pytest
import os
import geopandas as gpd
from shapely.geometry import LineString
from app.gis_integration import services
import pandas as pd

@pytest.fixture
def sample_trackways():
    """Sample trackways data for testing."""
    return {
        1: {
            "length": 10.0,
            "average_speed": 2.0,
            "points": pd.DataFrame({
                'x': [0, 1, 2],
                'y': [0, 1, 2],
                'time': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:00:01', '2023-01-01 00:00:02'])
            }).to_dict('records')
        },
        2: {
            "length": 5.0,
            "average_speed": 1.0,
            "points": pd.DataFrame({
                'x': [5, 6],
                'y': [5, 6],
                'time': pd.to_datetime(['2023-01-01 00:01:00', '2023-01-01 00:01:01'])
            }).to_dict('records')
        }
    }

@pytest.fixture
def sample_manual_gdf():
    """Sample manual trackways GeoDataFrame."""
    lines = [LineString([(0, 0), (1, 1), (2, 2)]), LineString([(10, 10), (11, 11)])]
    return gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")

def test_import_geojson(tmp_path):
    """Test importing a GeoJSON file."""
    filepath = tmp_path / "test.geojson"
    gdf = gpd.GeoDataFrame(geometry=[LineString([(0,0), (1,1)])], crs="EPSG:4326")
    gdf.to_file(filepath, driver='GeoJSON')

    imported_gdf = services.import_gis_data(str(filepath))
    assert isinstance(imported_gdf, gpd.GeoDataFrame)
    assert len(imported_gdf) == 1
    assert imported_gdf.crs == "EPSG:4326"

def test_export_trackways_geojson(tmp_path, sample_trackways):
    """Test exporting trackways to GeoJSON."""
    output_path = tmp_path / "exported_trackways.geojson"
    services.export_trackways(sample_trackways, 'geojson', str(output_path))

    assert output_path.exists()
    gdf = gpd.read_file(output_path)
    assert len(gdf) == 2
    assert 'trackway_id' in gdf.columns

def test_calculate_similarity(sample_trackways, sample_manual_gdf):
    """Test similarity calculation."""
    ai_geometries = [LineString([(p['x'], p['y']) for p in data['points']]) for data in sample_trackways.values()]
    ai_gdf = gpd.GeoDataFrame(geometry=ai_geometries, crs="EPSG:4326")
    ai_gdf['trackway_id'] = list(sample_trackways.keys())

    metrics = services.calculate_similarity(ai_gdf, sample_manual_gdf)

    assert 'overlap_percentage' in metrics
    assert 'average_offset' in metrics
    assert 'detection_completeness' in metrics
    assert metrics['detection_completeness'] == 50.0  # One of two manual trackways is detected
    assert len(metrics['matches']) == 1

def test_visualize_comparison(tmp_path, sample_trackways, sample_manual_gdf):
    """Test visualization generation."""
    output_path = tmp_path / "map.html"
    ai_geometries = [LineString([(p['x'], p['y']) for p in data['points']]) for data in sample_trackways.values()]
    ai_gdf = gpd.GeoDataFrame(geometry=ai_geometries, crs="EPSG:4326")

    services.visualize_comparison(ai_gdf, sample_manual_gdf, output_path=str(output_path))

    assert output_path.exists()
    with open(output_path, 'r') as f:
        content = f.read()
        assert "folium" in content

def test_generate_report(tmp_path):
    """Test report generation."""
    output_path = tmp_path / "report.txt"
    metrics = {
        "overlap_percentage": 50.0,
        "average_offset": 0.1,
        "detection_completeness": 100.0,
        "matches": [{'ai_trackway_id': 1, 'manual_trackway_id': 0, 'offset': 0.1}]
    }
    services.generate_report(metrics, 10.0, 100.0, str(output_path))

    assert output_path.exists()
    with open(output_path, 'r') as f:
        content = f.read()
        assert "Efficiency Gain: 10.00x" in content
