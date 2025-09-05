import pytest
from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io
import os
from app.config import settings

client = TestClient(app)

def test_ingest_jpg_success(create_dummy_image):
    """
    Test successful ingestion of a JPG file.
    """
    image_bytes = create_dummy_image("jpeg", 100, 100)
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Data ingested successfully"
    assert "metadata" in response.json()
    metadata = response.json()["metadata"]
    assert os.path.exists(metadata["file_path"])


def test_ingest_png_success(create_dummy_image):
    """
    Test successful ingestion of a PNG file.
    """
    image_bytes = create_dummy_image("png", 100, 100)
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Data ingested successfully"

def test_ingest_tiff_success(create_dummy_image):
    """
    Test successful ingestion of a TIFF file.
    """
    image_bytes = create_dummy_image("tiff", 100, 100)
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test.tiff", image_bytes, "image/tiff")},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Data ingested successfully"

def test_ingest_invalid_format():
    """
    Test ingestion of a file with an invalid format.
    """
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test.txt", b"some text", "text/plain")},
    )
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

def test_ingest_file_too_large(create_dummy_image):
    """
    Test ingestion of a file that is too large.
    """
    settings.max_file_size = 10
    image_bytes = create_dummy_image("jpeg", 100, 100)
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "File size exceeds the limit" in response.json()["detail"]
    settings.max_file_size = 10 * 1024 * 1024 # reset to default


def test_ingest_resolution_too_high(create_dummy_image):
    """
    Test ingestion of an image with a resolution that is too high.
    """
    settings.max_resolution = (10, 10)
    image_bytes = create_dummy_image("jpeg", 100, 100)
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "exceeds the limit of 10x10 pixels" in response.json()["detail"]
    settings.max_resolution = (4096, 4096) # reset to default

def test_ingest_and_process_success(create_dummy_image):
    """
    Test successful ingestion and processing of an image.
    """
    image_bytes = create_dummy_image("jpeg", 200, 200)
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test_process.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert "processed_file_path" in metadata
    assert "augmentations_applied" in metadata

    processed_path = metadata["processed_file_path"]
    assert os.path.exists(processed_path)

    # Verify the processed image
    processed_image = Image.open(processed_path)
    assert processed_image.size == settings.normalized_size
