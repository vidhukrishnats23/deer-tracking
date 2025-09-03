from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
from PIL import Image
import io
from . import services
from pydantic import BaseModel

router = APIRouter()

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str

class PredictionResult(BaseModel):
    filename: str
    predictions: List[BoundingBox]

@router.post("/predict/", response_model=List[PredictionResult])
async def predict_image(
    files: List[UploadFile] = File(...),
    generate_annotated_image: bool = False,
):
    """
    Accept one or more images and return bounding box predictions.
    Optionally, generate and save annotated images.
    """
    results = []
    for file in files:
        try:
            image = Image.open(io.BytesIO(await file.read()))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

        prediction = services.predict(image, file.filename, save=generate_annotated_image)

        # Extract bounding boxes, scores, and labels
        bboxes = []
        for box in prediction[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = box.conf[0].item()
            label = prediction[0].names[int(box.cls[0].item())]
            bboxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, score=score, label=label))

        results.append(PredictionResult(filename=file.filename, predictions=bboxes))

    return results
