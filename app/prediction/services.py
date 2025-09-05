from ultralytics import YOLO
from app.config import settings
from PIL import Image, ImageDraw
import os
import glob
import json
from datetime import datetime
from app.logger import logger

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

model_path = get_latest_model_path()
model = YOLO(model_path)

def _draw_bounding_boxes(image: Image.Image, predictions):
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

def predict(image: Image.Image, filename: str, save: bool = False):
    """
    Run prediction on a single image.
    Optionally save the annotated image and prediction data.
    """
    logger.info(f"Running prediction on {filename}")
    results = model(image)
    logger.info(f"Prediction complete for {filename}")

    if save:
        try:
            # Create a unique directory for each prediction
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prediction_dir = os.path.join("predictions", f"{filename}_{timestamp}")
            os.makedirs(prediction_dir, exist_ok=True)
            logger.info(f"Saving prediction results to {prediction_dir}")

            # Save annotated image
            annotated_image = _draw_bounding_boxes(image.copy(), results)
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
