import os
import yaml
from fastapi import FastAPI
from app.config import settings
from app.ingestion.router import router as ingestion_router
from app.annotation.router import router as annotation_router
from app.prediction.router import router as prediction_router
from app.trackways.router import router as trackways_router
from app.gis_integration.router import router as gis_integration_router
from app.logger import logger
from app.prediction.services import get_latest_model_path

app = FastAPI()


@app.on_event("startup")
def startup_event():
    """
    Startup event handler.
    Creates required directories, default data configuration file,
    and checks for model availability.
    """
    # Check for model availability
    try:
        model_path = get_latest_model_path()
        logger.info(f"Using model: {model_path}")
    except FileNotFoundError as e:
        logger.critical(str(e))
        raise e

    # Create required directories
    required_dirs = [
        settings.upload_dir,
        settings.processed_dir,
        settings.labels_dir,
        "reports",
        "data/train/images",
        "data/train/labels",
        "data/valid/images",
        "data/valid/labels",
    ]
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)

    # Create data/data.yaml if it doesn't exist
    if not os.path.exists(settings.data_config):
        data_config = {
            "train": "../data/train/images",
            "val": "../data/valid/images",
            "nc": 1,
            "names": ["deer"],
        }
        with open(settings.data_config, "w") as f:
            yaml.dump(data_config, f, default_flow_style=False)


app.include_router(ingestion_router, prefix="/api/v1")
app.include_router(annotation_router, prefix="/api/v1")
app.include_router(prediction_router, prefix="/api/v1")
app.include_router(trackways_router, prefix="/api/v1")
app.include_router(gis_integration_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health/")
def health_check():
    return {"status": "ok"}
