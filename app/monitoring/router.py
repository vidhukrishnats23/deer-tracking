from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from . import services
from app.logger import logger
from app.config import settings
import os

router = APIRouter()

@router.post("/monitoring/temporal_analysis", tags=["Monitoring"])
def temporal_analysis_endpoint(
    start_date1: str = Query(..., description="Start date for the first period (YYYY-MM-DD)"),
    end_date1: str = Query(..., description="End date for the first period (YYYY-MM-DD)"),
    start_date2: str = Query(..., description="Start date for the second period (YYYY-MM-DD)"),
    end_date2: str = Query(..., description="End date for the second period (YYYY-MM-DD)"),
    buffer_distance: float = Query(10.0, description="Buffer distance in meters for matching trackways."),
    cell_size: float = Query(10.0, description="Cell size in meters for the output intensity map."),
):
    """
    Triggers a temporal analysis of deer trackways between two periods.
    Returns paths to the generated report and map files.
    """
    try:
        logger.info(f"Received request for temporal analysis.")
        results = services.temporal_analysis(
            start_date1, end_date1, start_date2, end_date2, buffer_distance, cell_size
        )
        if results is None:
            raise HTTPException(status_code=500, detail="Error during temporal analysis.")

        # Make paths relative for the response
        if results.get("report_path"):
            results["report_path"] = os.path.basename(results["report_path"])
        if results.get("visualization_map_path"):
            results["visualization_map_path"] = os.path.basename(results["visualization_map_path"])
        if results.get("intensity_map_path"):
            results["intensity_map_path"] = os.path.basename(results["intensity_map_path"])

        return results
    except Exception as e:
        logger.error(f"Error in temporal analysis endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/report", tags=["Monitoring"])
def get_report(filename: str = Query(..., description="The filename of the report to retrieve.")):
    """
    Retrieves a specific HTML report file from the 'reports' directory.
    """
    reports_dir = "reports"
    file_path = os.path.join(reports_dir, filename)

    # Security check to prevent path traversal
    if not os.path.abspath(file_path).startswith(os.path.abspath(reports_dir)):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found.")

    return FileResponse(file_path, media_type='text/html', filename=filename)

@router.get("/monitoring/visualization", tags=["Monitoring"])
def get_visualization(filename: str = Query(..., description="The filename of the visualization map to retrieve.")):
    """
    Retrieves a specific HTML visualization map from the 'reports' directory.
    """
    reports_dir = "reports"
    file_path = os.path.join(reports_dir, filename)

    if not os.path.abspath(file_path).startswith(os.path.abspath(reports_dir)):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Map not found.")

    return FileResponse(file_path, media_type='text/html', filename=filename)

@router.get("/monitoring/intensity_map", tags=["Monitoring"])
def get_intensity_map(filename: str = Query(..., description="The filename of the intensity map to retrieve.")):
    """
    Retrieves a specific GeoTIFF intensity map from the processed data directory.
    """
    processed_dir = settings.processed_dir
    file_path = os.path.join(processed_dir, filename)

    if not os.path.abspath(file_path).startswith(os.path.abspath(processed_dir)):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Intensity map not found.")

    return FileResponse(file_path, media_type='image/tiff', filename=filename)
