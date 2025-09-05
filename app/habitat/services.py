from fastapi import UploadFile
from app.trackways.services import analyze_trackways
from app.gis_integration.services import get_habitat_areas, get_average_degradation_for_habitats
from app.config import settings
from collections import defaultdict
import random
from typing import Optional

async def classify_habitat(file: UploadFile, geotiff_path: Optional[str] = None):
    """
    Classifies the habitat in the given image.
    This is a mock implementation. In a real-world scenario, this would
    involve a machine learning model.
    """
    # Mock implementation
    habitat_types = ["forest", "grassland", "wetland", "urban"]
    habitat = random.choice(habitat_types)

    # In a real implementation, the GeoTIFF would be used for more context
    if geotiff_path:
        # For example, use the CRS or bounds to adjust classification
        pass

    return {
        "filename": file.filename,
        "predicted_habitat": habitat,
        "confidence": round(random.uniform(0.7, 0.99), 2),
        "geotiff_used": geotiff_path is not None
    }

def calculate_habitat_impact():
    """
    Calculates the density of deer trackways within different habitat types.
    """
    # 1. Get habitat areas
    habitat_areas = get_habitat_areas(settings.habitat_map_path)
    if not habitat_areas:
        return {"error": "Could not calculate habitat areas."}

    # 2. Get trackways
    trackways = analyze_trackways()
    if not trackways:
        return {"message": "No trackways found."}

    # 3. Count trackways per habitat type
    trackways_per_habitat = defaultdict(int)
    for trackway in trackways.values():
        habitat_type = trackway.get("habitat_type")
        if habitat_type is not None:
            trackways_per_habitat[habitat_type] += 1

    # 4. Calculate density
    habitat_impact = {}
    for habitat_type, count in trackways_per_habitat.items():
        area = habitat_areas.get(habitat_type)
        if area and area > 0:
            density = count / area # trackways per square meter
            habitat_impact[habitat_type] = {
                "trackway_count": count,
                "area_sqm": area,
                "density": density
            }

    return habitat_impact

def calculate_ecological_pressure():
    """
    Correlates trackway density with habitat degradation indicators.
    """
    # 1. Get habitat impact (trackway density)
    habitat_impact = calculate_habitat_impact()
    if "error" in habitat_impact or "message" in habitat_impact:
        return habitat_impact

    # 2. Get average degradation for each habitat
    avg_degradation = get_average_degradation_for_habitats(
        settings.habitat_map_path, settings.degradation_map_path
    )
    if not avg_degradation:
        return {"error": "Could not calculate average degradation."}

    # 3. Combine the data
    ecological_pressure = {}
    for habitat_type, impact_data in habitat_impact.items():
        # Ensure consistent key types
        degradation = avg_degradation.get(int(habitat_type))
        if degradation is not None:
            ecological_pressure[habitat_type] = {
                "trackway_density": impact_data["density"],
                "average_degradation": degradation
            }

    return ecological_pressure
