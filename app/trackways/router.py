from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Depends, Body
from typing import Optional
from . import services
from app.logger import logger
import os
from pydantic import BaseModel
from app.prediction.router import get_temp_geotiff_path


router = APIRouter()

class TrackwayAnalysisRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    geotiff: Optional[UploadFile] = None


@router.post("/trackways/analyze", tags=["Trackways"])
def analyze_trackways_endpoint(
    request: TrackwayAnalysisRequest = Body(...)
):
    """
    Trigger trackway analysis from existing detection data.
    """
    try:
        logger.info(f"Received request to analyze trackways from {request.start_date} to {request.end_date}")
        results = services.analyze_trackways(request.start_date, request.end_date)
        if results is None:
            return {"message": "No trackways found or an error occurred."}
        return results
    except Exception as e:
        logger.error(f"Error during trackway analysis: {e}")
        raise HTTPException(status_code=500, detail="Error during trackway analysis")

@router.post("/trackways/analyze_image", tags=["Trackways"])
async def analyze_trackways_from_image_endpoint(
    file: UploadFile = File(...),
    geotiff_path: Optional[str] = Depends(get_temp_geotiff_path)
):
    """
    Full workflow: detect, analyze trackways, and perform habitat analysis from a single image.
    """
    try:
        logger.info(f"Received request to analyze trackways from image {file.filename}")
        results = await services.analyze_trackways_from_image(file, geotiff_path)
        if results is None:
            return {"message": "No trackways found or an error occurred."}
        return results
    except Exception as e:
        logger.error(f"Error during trackway analysis from image: {e}")
        raise HTTPException(status_code=500, detail="Error during trackway analysis from image")


@router.post("/trackways/extract_features", tags=["Trackways"])
async def extract_features_endpoint(file: UploadFile = File(...)):
    """
    Extract linear features from an uploaded image.
    """
    try:
        image_bytes = await file.read()
        logger.info(f"Received request to extract features from {file.filename}")
        lines = services.extract_linear_features(image_bytes)

        if lines is None:
            raise HTTPException(status_code=500, detail="Error extracting features")

        return {"filename": file.filename, "features": lines}

    except Exception as e:
        logger.error(f"Error during feature extraction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during feature extraction: {str(e)}")
