from fastapi import APIRouter, File, UploadFile, HTTPException
from . import validation
from . import services
from app.logger import logger

router = APIRouter()

@router.post("/ingest")
async def ingest_data(file: UploadFile = File(...)):
    """
    Ingest a new image file.
    The file will be validated and saved to the raw data directory.
    """
    try:
        logger.info(f"Ingesting file: {file.filename}")
        error = validation.validate_file(file)
        if error:
            logger.warning(f"Validation error for file {file.filename}: {error}")
            raise HTTPException(status_code=400, detail=error)

        metadata = services.save_file(file)
        logger.info(f"Successfully ingested file: {file.filename}")
        return {"message": "Data ingested successfully", "metadata": metadata}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error ingesting file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error ingesting file {file.filename}")
