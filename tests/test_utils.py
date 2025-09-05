import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings
from app import cleanup
import os
import time
import shutil

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_and_teardown():
    """
    Fixture to set up test environment before each test
    and clean up after.
    """
    # Setup: ensure all required directories exist
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.processed_dir, exist_ok=True)
    os.makedirs(settings.project_name, exist_ok=True)

    yield

    # Teardown: clean up created files and directories
    if os.path.exists(settings.upload_dir):
        shutil.rmtree(settings.upload_dir)
    if os.path.exists(settings.processed_dir):
        shutil.rmtree(settings.processed_dir)
    if os.path.exists(settings.project_name):
        shutil.rmtree(settings.project_name)

def test_cleanup_processed_images():
    """
    Test that old processed images are removed by the cleanup utility.
    """
    # Create a dummy file with an old modification time
    old_file_path = os.path.join(settings.processed_dir, "old_file.jpg")
    with open(old_file_path, "w") as f:
        f.write("dummy")

    # Set modification time to be older than the retention period
    retention_period = settings.processed_images_retention_days * 86400
    old_time = time.time() - retention_period - 3600 # 1 hour older
    os.utime(old_file_path, (old_time, old_time))

    # Create a new file that should not be deleted
    new_file_path = os.path.join(settings.processed_dir, "new_file.jpg")
    with open(new_file_path, "w") as f:
        f.write("dummy")

    cleanup.cleanup_processed_images()

    assert not os.path.exists(old_file_path)
    assert os.path.exists(new_file_path)

def test_cleanup_training_artifacts():
    """
    Test that old training artifacts are removed by the cleanup utility.
    """
    # Create a dummy directory with an old modification time
    old_dir_path = os.path.join(settings.project_name, "old_run")
    os.makedirs(old_dir_path)

    # Set modification time to be older than the retention period
    retention_period = settings.training_artifacts_retention_days * 86400
    old_time = time.time() - retention_period - 3600 # 1 hour older
    os.utime(old_dir_path, (old_time, old_time))

    # Create a new directory that should not be deleted
    new_dir_path = os.path.join(settings.project_name, "new_run")
    os.makedirs(new_dir_path)

    cleanup.cleanup_training_artifacts()

    assert not os.path.exists(old_dir_path)
    assert os.path.exists(new_dir_path)

def test_ingest_corrupted_image():
    """
    Test ingestion of a corrupted image file.
    """
    # Create a corrupted image file (e.g., an incomplete file)
    corrupted_image_bytes = b"\xff\xd8\xff\xe0\x00\x10\x4a\x46"

    response = client.post(
        "/api/v1/ingest",
        files={"file": ("corrupted.jpg", corrupted_image_bytes, "image/jpeg")},
    )

    assert response.status_code == 400
    assert "Image file is corrupted" in response.json()["detail"]

def test_atomic_file_upload(create_dummy_image):
    """
    Test that file upload is atomic.
    This is a simplified test to ensure the file is created.
    A more rigorous test would require simulating concurrent requests.
    """
    image_bytes = create_dummy_image("jpeg", 100, 100)

    response = client.post(
        "/api/v1/ingest",
        files={"file": ("atomic_test.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert os.path.exists(metadata["file_path"])

    # Check that no temporary files are left
    temp_files = [f for f in os.listdir(settings.upload_dir) if f.startswith("temp_")]
    assert len(temp_files) == 0
