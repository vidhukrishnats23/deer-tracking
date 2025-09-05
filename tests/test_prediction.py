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

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.prediction.services.YOLO", MockYOLO)
    monkeypatch.setattr("app.prediction.services.get_latest_model_path", lambda: "dummy_model.pt")
    return TestClient(app)

def create_dummy_image(filename="test.jpg"):
    """
    Create a dummy image file.
    """
    image = Image.new('RGB', (100, 100), color = 'black')
    image.save(filename)
    return filename

def test_predict_single_image(client):
    """
    Test prediction on a single image.
    """
    image_path = create_dummy_image()
    with open(image_path, "rb") as f:
        response = client.post("/api/v1/predict/", files={"files": (image_path, f, "image/jpeg")})

    os.remove(image_path)

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert "filename" in data[0]
    assert "predictions" in data[0]
    assert data[0]["filename"] == image_path

def test_predict_multiple_images(client):
    """
    Test prediction on multiple images.
    """
    image_path1 = create_dummy_image("test1.jpg")
    image_path2 = create_dummy_image("test2.jpg")

    with open(image_path1, "rb") as f1, open(image_path2, "rb") as f2:
        files = [
            ("files", (image_path1, f1, "image/jpeg")),
            ("files", (image_path2, f2, "image/jpeg")),
        ]
        response = client.post("/api/v1/predict/", files=files)

    os.remove(image_path1)
    os.remove(image_path2)

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["filename"] == image_path1
    assert data[1]["filename"] == image_path2
