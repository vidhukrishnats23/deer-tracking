from fastapi import APIRouter, HTTPException, Body
from typing import List
from . import services
from app.logger import logger

router = APIRouter()

@router.post("/mosaic")
async def create_mosaic_endpoint(
    image_paths: List[str] = Body(..., description="List of image filenames to mosaic."),
    output_filename: str = Body(..., description="Filename for the output mosaic.")
):
    """
    Create a mosaic from a list of images.
    """
    try:
        logger.info(f"Creating mosaic for images: {image_paths}")
        output_path = services.create_mosaic(image_paths, output_filename)
        logger.info(f"Successfully created mosaic: {output_path}")
        return {"message": "Mosaic created successfully", "output_path": output_path}
    except Exception as e:
        logger.error(f"Error creating mosaic: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating mosaic: {e}")
