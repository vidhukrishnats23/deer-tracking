import pandas as pd
from sklearn.cluster import DBSCAN
import os
from app.logger import logger
from app.trackways.validation import is_biologically_plausible
from typing import Optional
import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from app.gis_integration.services import get_habitat_type_for_coord
from app.config import settings
from app.geospatial.services import calculate_morans_i, calculate_distance_to_nearest_feature

detections_file = "detections/detections.csv"

def _calc_displacement(trj):
    return np.sqrt(np.power(trj.x.shift(1) - trj.x, 2) + np.power(trj.y.shift(1) - trj.y, 2))

def analyze_trackways(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Analyzes deer trackways from detection data.
    Optionally, filter by a time window.
    """
    if not os.path.exists(detections_file):
        logger.warning("Detections file not found. No trackways to analyze.")
        return None

    # Read detection points
    df = pd.read_csv(detections_file)
    if df.empty:
        logger.info("Detections file is empty. No trackways to analyze.")
        return None

    # Convert timestamp to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter by date range if provided
    if start_date:
        df = df[df['timestamp'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['timestamp'] <= pd.to_datetime(end_date)]

    if df.empty:
        logger.info("No detections found in the specified time window.")
        return None

    # Cluster detection points to identify potential trackways
    # Using DBSCAN which is good for this kind of data.
    # The parameters eps and min_samples will need tuning.
    coords = df[['x_center', 'y_center']].values
    db = DBSCAN(eps=50, min_samples=5).fit(coords)
    labels = db.labels_

    # Add cluster labels to the dataframe
    df['cluster'] = labels

    # Filter out noise (points not assigned to any cluster)
    trackway_points = df[df['cluster'] != -1]

    if trackway_points.empty:
        logger.info("No significant trackways found after clustering.")
        return None

    # Analyze each cluster as a potential trackway
    results = {}
    for cluster_id in trackway_points['cluster'].unique():
        cluster_points = trackway_points[trackway_points['cluster'] == cluster_id].copy()

        # Sort points by timestamp to create a trajectory
        cluster_points = cluster_points.sort_values(by='timestamp')

        # Rename columns
        cluster_points.rename(columns={'x_center': 'x', 'y_center': 'y', 'timestamp': 'time'}, inplace=True)

        # Perform some basic analysis
        displacement = _calc_displacement(cluster_points)
        path_length = displacement.sum()
        time_diff = cluster_points['time'].diff().dt.total_seconds()
        path_speed = (displacement / time_diff).fillna(0)

        # Validate the trackway
        if not is_biologically_plausible(path_length, path_speed, cluster_points):
            continue

        # Get habitat type for the trackway's centroid
        centroid_x = cluster_points['x'].mean()
        centroid_y = cluster_points['y'].mean()
        habitat_type = get_habitat_type_for_coord(centroid_x, centroid_y, settings.habitat_map_path)

        # Calculate Moran's I for speed
        cluster_points['speed'] = path_speed
        gdf_points = gpd.GeoDataFrame(
            cluster_points,
            geometry=gpd.points_from_xy(cluster_points.x, cluster_points.y)
        )
        morans_i, morans_p = calculate_morans_i(gdf_points, 'speed')


        # Calculate confidence score stats
        confidence_mean = cluster_points['score'].mean()
        confidence_std = cluster_points['score'].std()

        results[int(cluster_id)] = {
            "length": path_length,
            "average_speed": path_speed.mean() if not path_speed.empty else 0,
            "points": cluster_points.to_dict('records'),
            "habitat_type": habitat_type,
            "morans_i": morans_i,
            "morans_i_p_value": morans_p,
            "confidence_mean": confidence_mean,
            "confidence_std": confidence_std,
        }

    logger.info(f"Analyzed {len(results)} potential trackways after validation.")

    return results


from app.prediction.services import predict as run_prediction
from app.habitat.services import classify_habitat
from fastapi import UploadFile
from app.geospatial.utils import pixel_to_geo
from PIL import Image
import io
import geopandas as gpd
from shapely.geometry import Point, LineString
from datetime import datetime

def _extract_deer_points_from_predictions(predictions, geotiff_path):
    deer_points = []
    for box in predictions[0].boxes:
        label = predictions[0].names[int(box.cls[0].item())]
        if label == 'deer':
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            if geotiff_path:
                center_x, center_y = pixel_to_geo(center_x, center_y, geotiff_path)
            deer_points.append(Point(center_x, center_y))
    return deer_points

def _cluster_points_to_trackways(gdf):
    coords = np.array([[p.x, p.y] for p in gdf.geometry])
    db = DBSCAN(eps=0.1, min_samples=3).fit(coords)
    gdf['cluster'] = db.labels_
    return gdf[gdf['cluster'] != -1]

def _convert_clusters_to_linestrings(trackway_gdf):
    trackways = []
    for cluster_id, cluster_gdf in trackway_gdf.groupby('cluster'):
        if len(cluster_gdf) > 1:
            line = LineString(cluster_gdf.geometry.tolist())
            trackways.append({
                "type": "Feature",
                "geometry": line.__geo_interface__,
                "properties": {
                    "trackway_id": int(cluster_id),
                    "length": line.length,
                }
            })
    return trackways

def _add_edge_effect_analysis_to_trackways(trackways, trackway_gdf, image_bytes):
    linear_features = extract_linear_features(image_bytes)
    for trackway in trackways:
        trackway_id = trackway["properties"]["trackway_id"]
        cluster_gdf = trackway_gdf[trackway_gdf['cluster'] == trackway_id]
        distances = calculate_distance_to_nearest_feature(cluster_gdf, linear_features)
        if distances and all(d is not None for d in distances):
            trackway["properties"]["avg_dist_to_feature"] = np.mean(distances)
        else:
            trackway["properties"]["avg_dist_to_feature"] = None
    return trackways

def _generate_report(trackways):
    with open("reporting/scientific_report_template.md", "r") as f:
        report_template = f.read()

    report = report_template.replace("[Date]", datetime.now().strftime("%Y-%m-%d"))
    report = report.replace("[value]", str(len(trackways)), 1)

    avg_length = np.mean([t["properties"]["length"] for t in trackways]) if trackways else 0
    report = report.replace("[value]", f"{avg_length:.2f}", 1)

    with open("report.md", "w") as f:
        f.write(report)

    return "report.md"


async def analyze_trackways_from_image(file: UploadFile, geotiff_path: Optional[str] = None):
    """
    Full workflow from image to trackway analysis.
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    predictions = run_prediction(image, file.filename)

    deer_points = _extract_deer_points_from_predictions(predictions, geotiff_path)
    if not deer_points:
        return None

    gdf = gpd.GeoDataFrame(geometry=deer_points, crs="EPSG:4326" if geotiff_path else None)
    gdf['timestamp'] = pd.to_datetime(pd.Timestamp.now()) # Mock timestamp

    trackway_gdf = _cluster_points_to_trackways(gdf)
    if trackway_gdf.empty:
        return None

    trackways = _convert_clusters_to_linestrings(trackway_gdf)

    file.file.seek(0)
    image_bytes_for_features = file.file.read()
    trackways = _add_edge_effect_analysis_to_trackways(trackways, trackway_gdf, image_bytes_for_features)

    habitat_info = await classify_habitat(file, geotiff_path)
    report_path = _generate_report(trackways)

    return {
        "trackways": {
            "type": "FeatureCollection",
            "features": trackways
        },
        "habitat_info": habitat_info,
        "report_generated": report_path
    }


def extract_linear_features(image_bytes: bytes):
    """
    Extracts linear features from image bytes using Hough Line Transform.
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Could not decode image from bytes")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is not None:
            logger.info(f"Detected {len(lines)} lines in image")
            return lines.tolist()
        else:
            logger.info("No lines detected in image")
            return []

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None
