from ultralytics import YOLO
from app.config import settings
from PIL import Image
import os
import glob

def get_latest_model_path():
    """
    Get the path to the latest YOLO model.
    If a trained model is available, use it. Otherwise, use the base model.
    """
    project_dir = settings.project_name
    if not os.path.isdir(project_dir):
        return settings.yolo_model_path

    list_of_runs = glob.glob(os.path.join(project_dir, '*'))
    if not list_of_runs:
        return settings.yolo_model_path

    latest_run = max(list_of_runs, key=os.path.getctime)
    weights_path = os.path.join(latest_run, 'weights', 'best.pt')

    if os.path.exists(weights_path):
        return weights_path
    else:
        return settings.yolo_model_path

model_path = get_latest_model_path()
model = YOLO(model_path)

def predict(image: Image.Image):
    """
    Run prediction on a single image.
    """
    results = model(image)
    return results
