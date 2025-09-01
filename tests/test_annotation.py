import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
from app.config import settings

client = TestClient(app)

def create_dummy_annotation_file(content: str) -> bytes:
    """
    Create a dummy annotation file for testing.
    """
    return content.encode("utf-8")

def test_annotate_success():
    """
    Test successful annotation upload.
    """
    annotation_content = "0 0.5 0.5 0.1 0.1"
    annotation_bytes = create_dummy_annotation_file(annotation_content)

    # Clean up any existing label file
    label_path = os.path.join(settings.labels_dir, "test.txt")
    if os.path.exists(label_path):
        os.remove(label_path)

    response = client.post(
        "/api/v1/annotate",
        files={"file": ("test.txt", annotation_bytes, "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Annotation data ingested successfully"
    assert "file_path" in response.json()
    file_path = response.json()["file_path"]
    assert os.path.exists(file_path)

    # Verify the content of the saved file
    with open(file_path, "r") as f:
        assert f.read() == annotation_content

def test_annotate_invalid_format():
    """
    Test annotation upload with invalid format.
    """
    annotation_content = "0 0.5 0.5 0.1" # Missing one value
    annotation_bytes = create_dummy_annotation_file(annotation_content)
    response = client.post(
        "/api/v1/annotate",
        files={"file": ("test_invalid.txt", annotation_bytes, "text/plain")},
    )
    assert response.status_code == 400
    assert "Invalid number of parts on line 1" in response.json()["detail"]

def test_annotate_out_of_range():
    """
    Test annotation upload with out-of-range values.
    """
    annotation_content = "0 1.5 0.5 0.1 0.1" # x_center > 1
    annotation_bytes = create_dummy_annotation_file(annotation_content)
    response = client.post(
        "/api/v1/annotate",
        files={"file": ("test_out_of_range.txt", annotation_bytes, "text/plain")},
    )
    assert response.status_code == 400
    assert "Values out of range on line 1" in response.json()["detail"]
