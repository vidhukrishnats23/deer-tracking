from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Body
from fastapi.responses import FileResponse
from . import services
from app.logger import logger
import os
import time
from app.trackways.services import analyze_trackways
import geopandas as gpd
from shapely.geometry import LineString
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class ComparisonRequest(BaseModel):
    manual_gis_file: UploadFile
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@router.post("/gis/import", tags=["GIS Integration"])
async def import_gis_endpoint(file: UploadFile = File(...)):
    """
    Import GIS data (Shapefile, GeoJSON).
    """
    temp_dir = "temp_gis"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        gdf = services.import_gis_data(temp_path)
        return {"filename": file.filename, "features_count": len(gdf)}
    except Exception as e:
        logger.error(f"Error during GIS import: {e}")
        raise HTTPException(status_code=500, detail=f"Error importing GIS data: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/compare/manual-vs-ai", tags=["GIS Integration"])
async def compare_manual_vs_ai_endpoint(
    request: ComparisonRequest = Body(...)
):
    """
    Compare AI-detected trackways with manual GIS data.
    """
    return await services.compare_manual_vs_ai(
        manual_gis_file=request.manual_gis_file,
        start_date=request.start_date,
        end_date=request.end_date
    )

class ExportRequest(BaseModel):
    format: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@router.post("/export/gis", tags=["GIS Integration"])
async def export_gis_endpoint(
    request: ExportRequest = Body(...)
):
    """
    Export AI-detected trackways to a GIS format.
    """
    return await services.export_gis(
        format=request.format,
        start_date=request.start_date,
        end_date=request.end_date
    )

@router.post("/gis/visualize", tags=["GIS Integration"])
async def visualize_comparison_endpoint(
    manual_gis_file: UploadFile = File(...),
    start_date: str = Form(None),
    end_date: str = Form(None),
    imagery_file: UploadFile = File(None),
):
    """
    Generate a visualization map comparing AI and manual trackways.
    """
    temp_dir = "temp_gis"
    os.makedirs(temp_dir, exist_ok=True)
    manual_gis_path = os.path.join(temp_dir, manual_gis_file.filename)
    imagery_path = None
    if imagery_file:
        imagery_path = os.path.join(temp_dir, imagery_file.filename)

    try:
        with open(manual_gis_path, "wb") as buffer:
            buffer.write(await manual_gis_file.read())
        if imagery_file:
            with open(imagery_path, "wb") as buffer:
                buffer.write(await imagery_file.read())

        # Get AI trackways and convert to GDF
        ai_trackways = analyze_trackways(start_date, end_date)
        if not ai_trackways:
            raise HTTPException(status_code=404, detail="No AI trackways found.")
        geometries = [LineString([(p['x'], p['y']) for p in data['points']]) for data in ai_trackways.values() if len(data['points']) >= 2]
        ai_gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
        ai_gdf['trackway_id'] = list(ai_trackways.keys())

        # Import manual GIS data
        manual_gdf = services.import_gis_data(manual_gis_path)

        # Generate map
        output_path = f"comparison_map_{int(time.time())}.html"
        services.visualize_comparison(ai_gdf, manual_gdf, imagery_path, output_path)

        return FileResponse(output_path, media_type='text/html')

    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Error during visualization: {str(e)}")
    finally:
        if os.path.exists(manual_gis_path):
            os.remove(manual_gis_path)
        if imagery_path and os.path.exists(imagery_path):
            os.remove(imagery_path)
