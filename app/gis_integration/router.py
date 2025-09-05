from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse
from . import services
from app.logger import logger
import os
import time
from app.trackways.services import analyze_trackways
import geopandas as gpd
from shapely.geometry import LineString

router = APIRouter()

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

@router.post("/gis/compare", tags=["GIS Integration"])
async def compare_gis_endpoint(
    manual_gis_file: UploadFile = File(...),
    start_date: str = Form(None),
    end_date: str = Form(None)
):
    """
    Compare AI-detected trackways with manual GIS data.
    """
    temp_dir = "temp_gis"
    os.makedirs(temp_dir, exist_ok=True)
    manual_gis_path = os.path.join(temp_dir, manual_gis_file.filename)

    try:
        with open(manual_gis_path, "wb") as buffer:
            buffer.write(await manual_gis_file.read())

        # 1. Get AI trackways
        ai_trackways = analyze_trackways(start_date, end_date)
        if not ai_trackways:
            raise HTTPException(status_code=404, detail="No AI trackways found for the given dates.")

        # 2. Convert AI trackways to GeoDataFrame
        geometries = [LineString([(p['x'], p['y']) for p in data['points']]) for data in ai_trackways.values() if len(data['points']) >= 2]
        ai_gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
        ai_gdf['trackway_id'] = list(ai_trackways.keys())


        # 3. Import manual GIS data
        manual_gdf = services.import_gis_data(manual_gis_path)

        # 4. Calculate similarity
        metrics = services.calculate_similarity(ai_gdf, manual_gdf)
        return metrics

    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Error during comparison: {str(e)}")
    finally:
        if os.path.exists(manual_gis_path):
            os.remove(manual_gis_path)

@router.post("/gis/export", tags=["GIS Integration"])
async def export_trackways_endpoint(
    format: str = Form(...),
    start_date: str = Form(None),
    end_date: str = Form(None)
):
    """
    Export AI-detected trackways to a GIS format.
    """
    try:
        # Get AI trackways
        ai_trackways = analyze_trackways(start_date, end_date)
        if not ai_trackways:
            raise HTTPException(status_code=404, detail="No AI trackways found for the given dates.")

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"ai_trackways_{int(time.time())}"

        ext = ""
        if format.lower() == 'shapefile':
            ext = ".shp"
        elif format.lower() == 'geojson':
            ext = ".geojson"
        elif format.lower() == 'kml':
            ext = ".kml"
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")

        output_path = os.path.join(output_dir, filename + ext)

        services.export_trackways(ai_trackways, format, output_path)

        return FileResponse(output_path, media_type='application/octet-stream', filename=filename+ext)

    except Exception as e:
        logger.error(f"Error during export: {e}")
        raise HTTPException(status_code=500, detail=f"Error during export: {str(e)}")

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
