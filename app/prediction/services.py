from ultralytics import YOLO
from app.config import settings
from PIL import Image, ImageDraw
import os
import glob
import json
from datetime import datetime
from app.logger import logger
import pandas as pd


def get_latest_model_path():
    """
    Get the path to the latest YOLO model.
    If a trained model is available, use it. Otherwise, use the base model.
    """
    project_dir = settings.project_name
    if os.path.isdir(project_dir):
        list_of_runs = glob.glob(os.path.join(project_dir, '*'))
        if list_of_runs:
            latest_run = max(list_of_runs, key=os.path.getctime)
            weights_path = os.path.join(latest_run, 'weights', 'best.pt')
            if os.path.exists(weights_path):
                logger.info(f"Found trained model at {weights_path}")
                return weights_path

    # If no trained model is found, check if the base model exists
    if os.path.exists(settings.yolo_model_path):
        logger.info(f"Using base model at {settings.yolo_model_path}")
        return settings.yolo_model_path
    else:
        raise FileNotFoundError(
            f"Model file not found at {settings.yolo_model_path}. "
            "Please ensure the model is available or run the training pipeline."
        )

model = None

def get_model():
    """
    Load the YOLO model.
    """
    global model
    if model is None:
        model_path = get_latest_model_path()
        model = YOLO(model_path)
    return model

def _draw_bounding_boxes(image: Image.Image, predictions, model):
    """
    Draw bounding boxes on an image.
    """
    draw = ImageDraw.Draw(image)
    for box in predictions[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = model.names[int(box.cls[0].item())]
        confidence = box.conf[0].item()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{label} ({confidence:.2f})", fill="red")
    return image


def _save_detection_points(filename: str, predictions, model):
    """
    Save detection points to a CSV file.
    """
    detections_dir = "detections"
    os.makedirs(detections_dir, exist_ok=True)
    detections_file = os.path.join(detections_dir, "detections.csv")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_rows = []
    for box in predictions[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        score = box.conf[0].item()
        label = model.names[int(box.cls[0].item())]
        if label == 'deer':
            new_rows.append([timestamp, filename, x_center, y_center, score, label])

    if not new_rows:
        return

    df = pd.DataFrame(new_rows, columns=["timestamp", "filename", "x_center", "y_center", "score", "label"])

    try:
        if not os.path.exists(detections_file):
            df.to_csv(detections_file, index=False)
        else:
            df.to_csv(detections_file, mode='a', header=False, index=False)
        logger.info(f"Saved {len(new_rows)} detection points to {detections_file}")
    except Exception as e:
        logger.error(f"Error saving detection points to {detections_file}: {e}")


def predict(image: Image.Image, filename: str, save: bool = False):
    """
    Run prediction on a single image.
    Optionally save the annotated image and prediction data.
    """
    model = get_model()
    results = model(image)

    # Save deer detection points for trackway analysis
    _save_detection_points(filename, results, model)

    if save:
        try:
            # Create a unique directory for each prediction
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prediction_dir = os.path.join("predictions", f"{filename}_{timestamp}")
            os.makedirs(prediction_dir, exist_ok=True)
            logger.info(f"Saving prediction results to {prediction_dir}")

            # Save annotated image
            annotated_image = _draw_bounding_boxes(image.copy(), results, model)
            annotated_image_path = os.path.join(prediction_dir, f"annotated_{filename}")
            annotated_image.save(annotated_image_path)

            # Save prediction data
            prediction_data = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = box.conf[0].item()
                label = model.names[int(box.cls[0].item())]
                prediction_data.append({"box": [x1, y1, x2, y2], "score": score, "label": label})

            json_path = os.path.join(prediction_dir, "predictions.json")
            with open(json_path, "w") as f:
                json.dump(prediction_data, f, indent=4)
            logger.info(f"Successfully saved prediction results for {filename}")
        except Exception as e:
            logger.error(f"Error saving prediction results for {filename}: {e}")
            # We don't re-raise the exception here because the prediction itself was successful.
            # The client has the prediction results, even if saving failed.

    return results

from fastapi import UploadFile
import io

async def predict_stream(image_bytes: io.BytesIO, filename: str, save: bool = False):
    """
    Run prediction on a single image from a BytesIO object.
    This is designed for use with streaming responses.
    """
    image = Image.open(image_bytes)
    return predict(image, filename, save)
