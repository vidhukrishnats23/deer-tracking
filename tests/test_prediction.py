import pytest
from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io
import os
from app.config import settings


class MockTensor:
    def __init__(self, data):
        self.data = data
    def tolist(self):
        return self.data
    def item(self):
        return self.data

class MockBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [MockTensor(xyxy)]
        self.cls = [MockTensor(cls)]
        self.conf = [MockTensor(conf)]

    @property
    def item(self):
        return self.conf[0]

class MockResult:
    def __init__(self, model):
        self.boxes = [MockBox([10, 10, 50, 50], 0, 0.9)]
        self.names = model.names

class MockYOLO:
    def __init__(self, model_path):
        self.model_path = model_path
        self.names = {0: 'deer'}

    def __call__(self, image):
        return [MockResult(self)]

@pytest.fixture(scope="session", autouse=True)
def mock_yolo_model(session_mocker):
    """
    Fixture to mock the YOLO model for all tests.
    """
    session_mocker.patch("app.prediction.services.YOLO", MockYOLO)
    session_mocker.patch(
        "app.prediction.services.get_latest_model_path", lambda: "dummy_model.pt"
    )

@pytest.fixture
def client():
    return TestClient(app)

def create_dummy_image(filename="test.jpg"):
    """
    Create a dummy image file.
    """
    image = Image.new('RGB', (100, 100), color = 'black')
    image.save(filename)
    return filename

import rasterio
from rasterio.transform import from_origin
import numpy as np

@pytest.fixture
def sample_geotiff(tmp_path):
    """Create a dummy GeoTIFF file for testing."""
    width = 100
    height = 100
    transform = from_origin(10, 40, 1, 1)

    data = np.ones((height, width), dtype=np.uint8)

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

    file_path = os.path.join(tmp_path, "test.tif")
    with rasterio.open(file_path, 'w', **meta) as dst:
        dst.write(data, 1)

    return file_path

def test_predict_image_with_geotiff(client, sample_geotiff):
    """Test the /predict endpoint with a GeoTIFF file."""
    image_path = create_dummy_image()
    with open(image_path, "rb") as f, open(sample_geotiff, "rb") as geotiff:
        files = {
            "file": (image_path, f, "image/jpeg"),
            "geotiff_file": (sample_geotiff, geotiff, "image/tiff"),
        }
        response = client.post("/api/v1/predict/", files=files)

    os.remove(image_path)

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data
    assert len(data["features"]) > 0
    feature = data["features"][0]
    assert feature["geometry"]["type"] == "Polygon"
    # Check if coordinates are geographic
    assert feature["geometry"]["coordinates"][0][0][0] != 10.0

def test_predict_image_stream(client):
    """Test the /predict/stream endpoint."""
    image_path = create_dummy_image()
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        response = client.post("/api/v1/predict/stream", files=files)

    os.remove(image_path)

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    # We can't easily parse the streaming response here,
    # but we can check if it starts and ends correctly.
    content = response.text
    assert content.startswith('{"type": "FeatureCollection", "features": [')
    assert content.endswith(']}')
