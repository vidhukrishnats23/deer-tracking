from fastapi import APIRouter, File, UploadFile, HTTPException
from . import services
from app.logger import logger

router = APIRouter()

@router.post("/annotate")
async def annotate_data(file: UploadFile = File(...)):
    """
    Ingest a new annotation file.
    The file will be validated and saved to the labels directory.
    """
    try:
        logger.info(f"Annotating file: {file.filename}")
        file_path = services.save_annotation(file)
        error = services.validate_annotation(file_path)
        if error:
            logger.warning(f"Validation error for annotation file {file.filename}: {error}")
            raise HTTPException(status_code=400, detail=error)

        logger.info(f"Successfully annotated file: {file.filename}")
        return {"message": "Annotation data ingested successfully", "file_path": file_path}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error annotating file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error annotating file {file.filename}")
