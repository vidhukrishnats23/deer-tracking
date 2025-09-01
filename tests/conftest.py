import pytest
import tempfile
import shutil
from app.config import settings

@pytest.fixture(autouse=True)
def override_settings():
    """
    Override settings for tests.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        original_upload_dir = settings.upload_dir
        original_metadata_log_file = settings.metadata_log_file
        settings.upload_dir = tmpdir
        settings.metadata_log_file = f"{tmpdir}/metadata.log"
        yield
        settings.upload_dir = original_upload_dir
        settings.metadata_log_file = original_metadata_log_file
