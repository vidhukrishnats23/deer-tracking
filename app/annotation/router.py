from fastapi import APIRouter, File, UploadFile, HTTPException
from . import services

router = APIRouter()

@router.post("/annotate")
async def annotate_data(file: UploadFile = File(...)):
    """
    Ingest a new annotation file.
    The file will be validated and saved to the labels directory.
    """
    file_path = services.save_annotation(file)
    error = services.validate_annotation(file_path)
    if error:
        raise HTTPException(status_code=400, detail=error)

    return {"message": "Annotation data ingested successfully", "file_path": file_path}
