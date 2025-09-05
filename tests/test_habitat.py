from fastapi.testclient import TestClient
from app.main import app
import pytest
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from app.config import settings
import os

client = TestClient(app)

@pytest.fixture
def setup_habitat_test_data(tmp_path, monkeypatch):
    """
    Set up dummy data for habitat tests.
    """
    # Create dummy detections.csv
    detections_file = tmp_path / "detections.csv"
    # A cluster of 5 points forming a line with length > 10
    detections_data = {
        "x_center": [0.5, 3.5, 6.5, 9.5, 12.5],
        "y_center": [1.5, 1.5, 1.5, 1.5, 1.5],
        "timestamp": pd.to_datetime([f"2023-01-01 12:0{i}:00" for i in range(5)]),
    }
    pd.DataFrame(detections_data).to_csv(detections_file, index=False)

    # Monkeypatch the path to the detections file in the trackways service
    monkeypatch.setattr("app.trackways.services.detections_file", str(detections_file))

    # Create dummy habitat map (20x2)
    habitat_map_path = tmp_path / "habitat_map.tif"
    # The trackway is mostly in habitat 1
    habitat_data = np.array([[1]*20, [3]*20], dtype=np.uint8)
    transform = from_origin(0, 2, 1, 1) # 1x1 pixel size, origin at top-left
    with rasterio.open(
        habitat_map_path, 'w', driver='GTiff', height=2, width=20, count=1,
        dtype=habitat_data.dtype, crs='+proj=latlong', transform=transform
    ) as dst:
        dst.write(habitat_data, 1)

    # Create dummy degradation map (20x2)
    degradation_map_path = tmp_path / "degradation_map.tif"
    degradation_data = np.array([[0.1]*20, [0.5]*20], dtype=np.float32)
    with rasterio.open(
        degradation_map_path, 'w', driver='GTiff', height=2, width=20, count=1,
        dtype=degradation_data.dtype, crs='+proj=latlong', transform=transform
    ) as dst:
        dst.write(degradation_data, 1)

    # Override settings
    monkeypatch.setattr(settings, "habitat_map_path", str(habitat_map_path))
    monkeypatch.setattr(settings, "degradation_map_path", str(degradation_map_path))


def test_classify_habitat_endpoint(create_dummy_image):
    """
    Test the /habitat/classify endpoint.
    """
    image_bytes = create_dummy_image("jpeg", 100, 100)
    response = client.post(
        "/api/v1/habitat/classify",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "test.jpg"
    assert response.json()["habitat"] == "unknown"


def test_impact_assessment_endpoint(setup_habitat_test_data):
    """
    Test the /habitat/impact_assessment endpoint.
    """
    response = client.get("/api/v1/habitat/impact_assessment")
    assert response.status_code == 200
    data = response.json()
    # The centroid of the trackway is (6.5, 1.5), which falls in habitat 1.
    # The area of habitat 1 is 20 sqm.
    # So we expect 1 trackway in habitat 1, and density 1/20 = 0.05.
    assert "1" in data
    assert data["1"]["trackway_count"] == 1
    assert data["1"]["area_sqm"] == 20.0
    assert data["1"]["density"] == pytest.approx(0.05)


def test_ecological_pressure_endpoint(setup_habitat_test_data):
    """
    Test the /habitat/ecological_pressure endpoint.
    """
    response = client.get("/api/v1/habitat/ecological_pressure")
    assert response.status_code == 200
    data = response.json()
    # The trackway is in habitat 1.
    # The average degradation for habitat 1 is 0.1.
    # The trackway density is 0.05.
    assert "1" in data
    assert data["1"]["trackway_density"] == pytest.approx(0.05)
    assert data["1"]["average_degradation"] == pytest.approx(0.1)
