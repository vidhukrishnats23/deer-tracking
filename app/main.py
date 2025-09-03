from fastapi import FastAPI
from app.ingestion.router import router as ingestion_router
from app.annotation.router import router as annotation_router
from app.prediction.router import router as prediction_router

app = FastAPI()

app.include_router(ingestion_router, prefix="/api/v1")
app.include_router(annotation_router, prefix="/api/v1")
app.include_router(prediction_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health/")
def health_check():
    return {"status": "ok"}
