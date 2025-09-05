from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from typing import Optional
from . import services
from app.logger import logger
import os

router = APIRouter()

@router.post("/trackways/analyze", tags=["Trackways"])
def analyze_trackways_endpoint(
    start_date: Optional[str] = Query(None, description="Start date for analysis (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for analysis (YYYY-MM-DD)"),
):
    """
    Trigger trackway analysis.
    """
    try:
        logger.info(f"Received request to analyze trackways from {start_date} to {end_date}")
        results = services.analyze_trackways(start_date, end_date)
        if results is None:
            return {"message": "No trackways found or an error occurred."}
        return results
    except Exception as e:
        logger.error(f"Error during trackway analysis: {e}")
        raise HTTPException(status_code=500, detail="Error during trackway analysis")

@router.post("/trackways/extract_features", tags=["Trackways"])
async def extract_features_endpoint(file: UploadFile = File(...)):
    """
    Extract linear features from an uploaded image.
    """
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        logger.info(f"Received request to extract features from {file.filename}")
        lines = services.extract_linear_features(temp_path)

        if lines is None:
            raise HTTPException(status_code=500, detail="Error extracting features")

        return {"filename": file.filename, "features": lines}

    except Exception as e:
        logger.error(f"Error during feature extraction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during feature extraction: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
