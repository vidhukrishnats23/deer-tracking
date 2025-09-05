import os
import shutil
import time
from app.config import settings
from app.logger import logger

def cleanup_processed_images():
    """
    Remove processed images older than the retention period.
    """
    logger.info("Starting cleanup of processed images...")
    now = time.time()
    retention_period = settings.processed_images_retention_days * 86400  # in seconds

    for filename in os.listdir(settings.processed_dir):
        file_path = os.path.join(settings.processed_dir, filename)
        if os.path.isfile(file_path):
            if now - os.path.getmtime(file_path) > retention_period:
                os.remove(file_path)
                logger.info(f"Removed old processed image: {file_path}")

def cleanup_training_artifacts():
    """
    Remove old training artifacts.
    """
    logger.info("Starting cleanup of training artifacts...")
    now = time.time()
    retention_period = settings.training_artifacts_retention_days * 86400  # in seconds

    project_path = settings.project_name
    if os.path.exists(project_path):
        for dirname in os.listdir(project_path):
            dir_path = os.path.join(project_path, dirname)
            if os.path.isdir(dir_path):
                if now - os.path.getmtime(dir_path) > retention_period:
                    shutil.rmtree(dir_path)
                    logger.info(f"Removed old training artifacts directory: {dir_path}")

def run_cleanup():
    """
    Run all cleanup tasks.
    """
    logger.info("Running all cleanup tasks...")
    cleanup_processed_images()
    cleanup_training_artifacts()
    logger.info("Cleanup tasks finished.")

if __name__ == "__main__":
    run_cleanup()
