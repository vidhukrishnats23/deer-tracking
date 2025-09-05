import pytest
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np
from fastapi.testclient import TestClient

from app.monitoring.services import (
    _trackways_to_gdf,
    _compare_trackways,
    _calculate_statistical_trends,
    temporal_analysis,
)
from app.main import app

client = TestClient(app)

# Fixture for sample trackway data
@pytest.fixture
def sample_trackways():
    trackways1 = {
        "1": {"points": [{"x": 0, "y": 0}, {"x": 10, "y": 10}], "length": 14.14, "average_speed": 1.0}, # Abandoned
        "2": {"points": [{"x": 20, "y": 20}, {"x": 30, "y": 30}], "length": 14.14, "average_speed": 1.2}, # Modified
    }
    trackways2 = {
        "3": {"points": [{"x": 21, "y": 21}, {"x": 31, "y": 31}], "length": 14.14, "average_speed": 1.3}, # Modified
        "4": {"points": [{"x": 50, "y": 50}, {"x": 60, "y": 60}], "length": 14.14, "average_speed": 1.1}, # New
    }
    return trackways1, trackways2

# Fixture for GeoDataFrames
@pytest.fixture
def sample_gdfs(sample_trackways):
    trackways1, trackways2 = sample_trackways
    gdf1 = _trackways_to_gdf(trackways1)
    gdf2 = _trackways_to_gdf(trackways2)
    return gdf1, gdf2

def test_trackways_to_gdf(sample_trackways):
    trackways1, _ = sample_trackways
    gdf = _trackways_to_gdf(trackways1)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 2
    assert 'trackway_id' in gdf.columns
    assert gdf.crs is not None

def test_compare_trackways(sample_gdfs):
    gdf1, gdf2 = sample_gdfs
    result = _compare_trackways(gdf1, gdf2, buffer_distance=5.0)
    assert set(result["abandoned"]) == {"1"}
    assert set(result["new"]) == {"4"}
    assert set(result["modified"]) == {"3"}

def test_compare_trackways_no_overlap(sample_gdfs):
    gdf1, gdf2 = sample_gdfs
    # With a small buffer, trackway 2 and 3 should not be matched
    result = _compare_trackways(gdf1, gdf2, buffer_distance=0.1)
    assert set(result["abandoned"]) == {"1", "2"}
    assert set(result["new"]) == {"3", "4"}
    assert not result["modified"]

def test_calculate_statistical_trends(sample_gdfs):
    gdf1, gdf2 = sample_gdfs
    stats = _calculate_statistical_trends(gdf1, gdf2)
    assert "length" in stats
    assert "avg_speed" in stats
    # With only one modified trackway, the stats for the other are based on one sample.
    # The test should check if the calculation runs without error and returns the expected structure.
    assert stats["avg_speed"]["p_value"] is not None
    assert stats["avg_speed"]["significant_change"] is not None

def test_calculate_statistical_trends_insufficient_data():
    empty_gdf = _trackways_to_gdf({})
    gdf1, _ = _trackways_to_gdf({"1": {"points": [{"x": 0, "y": 0}, {"x": 1, "y": 1}], "length": 1.4, "average_speed": 1.0}}), _trackways_to_gdf({})
    stats = _calculate_statistical_trends(gdf1, empty_gdf)
    assert stats["length"]["p_value"] is None

def test_temporal_analysis_endpoint(monkeypatch, sample_trackways):
    # Mock the analyze_trackways service to return controlled data
    trackways1, trackways2 = sample_trackways

    # This mock function will be called instead of the real analyze_trackways
    def mock_analyze_trackways(start_date=None, end_date=None):
        if "1" in start_date: # A simple way to distinguish calls
            return trackways1
        else:
            return trackways2

    monkeypatch.setattr("app.monitoring.services.analyze_trackways", mock_analyze_trackways)

    # Mock functions that create files
    monkeypatch.setattr("app.monitoring.services._generate_impact_intensity_map", lambda a,b,c: "mock_intensity_map.tif")
    monkeypatch.setattr("app.monitoring.services._visualize_temporal_changes", lambda a,b,c: "mock_viz_map.html")
    monkeypatch.setattr("app.monitoring.services._generate_monitoring_report", lambda a: "mock_report.html")

    response = client.post(
        "/api/v1/monitoring/temporal_analysis",
        params={
            "start_date1": "p1_start", "end_date1": "p1_end",
            "start_date2": "p2_start", "end_date2": "p2_end"
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert "change_summary" in data
    assert set(data["change_summary"]["abandoned"]) == {"1"}
    assert set(data["change_summary"]["new"]) == {"4"}

    assert "statistical_summary" in data
    assert "report_path" in data
    assert data["report_path"] == "mock_report.html"
