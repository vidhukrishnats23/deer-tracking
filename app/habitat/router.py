from fastapi import APIRouter, UploadFile, File
from app.habitat.services import classify_habitat, calculate_habitat_impact, calculate_ecological_pressure
from app.habitat.validation import validate_image

router = APIRouter(
    prefix="/habitat",
    tags=["habitat"],
)

@router.post("/classify")
async def classify_habitat_endpoint(file: UploadFile = File(...)):
    """
    Classify the habitat type from an image.
    """
    validate_image(file)
    result = await classify_habitat(file)
    return result

@router.get("/impact_assessment")
def get_habitat_impact_assessment():
    """
    Get the habitat impact assessment.
    """
    impact = calculate_habitat_impact()
    return impact

@router.get("/ecological_pressure")
def get_ecological_pressure_analysis():
    """
    Get the ecological pressure analysis.
    """
    pressure = calculate_ecological_pressure()
    return pressure
