import os
import sys
from ultralytics import YOLO

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from app.config import settings

def train():
    """
    Train the YOLO model with the settings specified in the config file.
    """
    # Load the YOLO model
    model = YOLO(settings.yolo_model)

    # Train the model
    model.train(
        data=settings.data_config,
        epochs=settings.epochs,
        batch=settings.batch_size,
        imgsz=settings.img_size,
        project=settings.project_name,
        name=settings.run_name
    )

    # The model is saved automatically by ultralytics in the project/name directory
    print(f"Model training complete. The model is saved in the '{settings.project_name}/{settings.run_name}' directory.")

if __name__ == "__main__":
    train()
