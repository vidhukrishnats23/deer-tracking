from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from typing import List, Optional
from PIL import Image
import io
from . import services
from pydantic import BaseModel
from app.logger import logger
from app.geospatial.utils import pixel_to_geo
import tempfile
import os
from fastapi.responses import StreamingResponse
import json

router = APIRouter()

class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: dict
    properties: dict

class GeoJSONFeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature]

def get_temp_geotiff_path(geotiff_file: UploadFile = File(None)):
    if geotiff_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(geotiff_file.file.read())
            return tmp.name
    return None

@router.post("/predict/", response_model=GeoJSONFeatureCollection)
async def predict_image(
    file: UploadFile = File(...),
    geotiff_path: Optional[str] = Depends(get_temp_geotiff_path),
    generate_annotated_image: bool = Form(False),
):
    """
    Accept an image and return bounding box predictions as GeoJSON.
    If a GeoTIFF is provided, the coordinates will be in the image's CRS.
    Otherwise, coordinates will be pixel values.
    """
    try:
        logger.info(f"Processing file: {file.filename}")
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        prediction = services.predict(image, file.filename, save=generate_annotated_image)

        features = []
        for box in prediction[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = box.conf[0].item()
            label = prediction[0].names[int(box.cls[0].item())]

            if geotiff_path:
                # Convert pixel coordinates to geographic coordinates
                geo_x1, geo_y1 = pixel_to_geo(x1, y1, geotiff_path)
                geo_x2, geo_y2 = pixel_to_geo(x2, y2, geotiff_path)
                # Create a GeoJSON Polygon
                geometry = {
                    "type": "Polygon",
                    "coordinates": [[
                        [geo_x1, geo_y1],
                        [geo_x2, geo_y1],
                        [geo_x2, geo_y2],
                        [geo_x1, geo_y2],
                        [geo_x1, geo_y1]
                    ]]
                }
            else:
                # Use pixel coordinates if no GeoTIFF is provided
                geometry = {
                    "type": "Polygon",
                    "coordinates": [[
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]
                    ]]
                }

            feature = GeoJSONFeature(
                geometry=geometry,
                properties={"score": score, "label": label, "filename": file.filename}
            )
            features.append(feature)

        logger.info(f"Successfully processed file: {file.filename}")
        return GeoJSONFeatureCollection(features=features)

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}")
    finally:
        if geotiff_path and os.path.exists(geotiff_path):
            os.remove(geotiff_path)

@router.post("/predict/stream", tags=["Prediction"])
async def predict_image_stream(
    file: UploadFile = File(...),
    geotiff_file: UploadFile = File(None),
    generate_annotated_image: bool = Form(False),
):
    """
    Accept an image and stream bounding box predictions as GeoJSON features.
    This is suitable for large images and continuous data streams.
    """
    geotiff_path = None
    if geotiff_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(geotiff_file.file.read())
            geotiff_path = tmp.name

    image_bytes = await file.read()
    file.file.close()


    async def generate():
        try:
            logger.info(f"Streaming processing for file: {file.filename}")

            # The predict service will now handle reading the file
            prediction = await services.predict_stream(io.BytesIO(image_bytes), file.filename, generate_annotated_image)

            yield '{"type": "FeatureCollection", "features": ['
            first = True
            for box in prediction[0].boxes:
                if not first:
                    yield ','
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = box.conf[0].item()
                label = prediction[0].names[int(box.cls[0].item())]

                if geotiff_path:
                    geo_x1, geo_y1 = pixel_to_geo(x1, y1, geotiff_path)
                    geo_x2, geo_y2 = pixel_to_geo(x2, y2, geotiff_path)
                    geometry = {"type": "Polygon", "coordinates": [[[geo_x1, geo_y1], [geo_x2, geo_y1], [geo_x2, geo_y2], [geo_x1, geo_y2], [geo_x1, geo_y1]]]}
                else:
                    geometry = {"type": "Polygon", "coordinates": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]]}

                feature = GeoJSONFeature(geometry=geometry, properties={"score": score, "label": label})
                yield json.dumps(feature.dict())
                first = False
            yield ']}'
            logger.info(f"Successfully streamed processing for file: {file.filename}")

        except Exception as e:
            logger.error(f"Error streaming file {file.filename}: {e}")
            # This part of the code won't be able to raise an HTTPException
            # as the response has already started streaming.
            # We can log the error and end the stream.
        finally:
            if geotiff_path and os.path.exists(geotiff_path):
                os.remove(geotiff_path)

    return StreamingResponse(generate(), media_type="application/json")
