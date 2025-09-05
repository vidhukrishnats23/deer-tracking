import pandas as pd
from sklearn.cluster import DBSCAN
import os
from app.logger import logger
from app.trackways.validation import is_biologically_plausible
from typing import Optional
import cv2
import numpy as np
from app.gis_integration.services import get_habitat_type_for_coord
from app.config import settings

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


        results[int(cluster_id)] = {
            "length": path_length,
            "average_speed": path_speed.mean() if not path_speed.empty else 0,
            "points": cluster_points.to_dict('records'),
            "habitat_type": habitat_type,
        }

    logger.info(f"Analyzed {len(results)} potential trackways after validation.")

    return results


def extract_linear_features(image_path: str):
    """
    Extracts linear features from an image using Hough Line Transform.
    """
    if not os.path.exists(image_path):
        logger.error(f"Image not found at {image_path}")
        return None

    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image at {image_path}")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is not None:
            logger.info(f"Detected {len(lines)} lines in {image_path}")
            return lines.tolist()
        else:
            logger.info(f"No lines detected in {image_path}")
            return []

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None
