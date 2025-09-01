from fastapi import APIRouter, File, UploadFile, HTTPException
from . import validation
from . import services

router = APIRouter()

@router.post("/ingest")
async def ingest_data(file: UploadFile = File(...)):
    """
    Ingest a new image file.
    The file will be validated and saved to the raw data directory.
    """
    error = validation.validate_file(file)
    if error:
        raise HTTPException(status_code=400, detail=error)

    metadata = services.save_file(file)
    services.log_metadata(metadata)

    return {"message": "Data ingested successfully", "metadata": metadata}
