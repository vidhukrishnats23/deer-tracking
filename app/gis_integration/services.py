import geopandas as gpd
from app.logger import logger

def import_gis_data(filepath: str):
    """
    Imports GIS data from a Shapefile or GeoJSON file.

    Args:
        filepath (str): The path to the GIS file.

    Returns:
        geopandas.GeoDataFrame: The imported GIS data as a GeoDataFrame.
    """
    try:
        logger.info(f"Importing GIS data from {filepath}")
        gdf = gpd.read_file(filepath)
        logger.info(f"Successfully imported {len(gdf)} features from {filepath}")
        return gdf
    except Exception as e:
        logger.error(f"Error importing GIS data from {filepath}: {e}")
        raise e

from shapely.geometry import LineString

def export_trackways(trackways: dict, format: str, output_path: str):
    """
    Exports AI-detected trackways to a GIS-compatible format.

    Args:
        trackways (dict): A dictionary of trackways, where each key is a trackway ID
                          and each value contains a list of points.
        format (str): The output format ('Shapefile', 'GeoJSON', 'KML').
        output_path (str): The path to save the output file.
    """
    try:
        logger.info(f"Exporting {len(trackways)} trackways to {output_path}")

        geometries = []
        for trackway_id, data in trackways.items():
            points = data.get("points", [])
            if len(points) < 2:
                continue

            # Ensure points are sorted by time if available
            if 'time' in points[0]:
                points.sort(key=lambda p: p['time'])

            line = LineString([(p['x'], p['y']) for p in points])
            geometries.append({
                'geometry': line,
                'properties': {
                    'trackway_id': trackway_id,
                    'length': data.get('length'),
                    'avg_speed': data.get('average_speed')
                }
            })

        if not geometries:
            logger.warning("No valid trackways to export.")
            return

        gdf = gpd.GeoDataFrame.from_features(geometries)

        # Set a generic CRS, this should be improved with actual data
        gdf.crs = "EPSG:4326"

        if format.lower() == 'shapefile':
            gdf.to_file(output_path, driver='ESRI Shapefile')
        elif format.lower() == 'geojson':
            gdf.to_file(output_path, driver='GeoJSON')
        elif format.lower() == 'kml':
            # KML driver requires fiona>=1.8.4
            gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
            gdf.to_file(output_path, driver='KML')
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Successfully exported trackways to {output_path}")

    except Exception as e:
        logger.error(f"Error exporting trackways: {e}")
        raise e

def calculate_similarity(ai_trackways_gdf: gpd.GeoDataFrame, manual_trackways_gdf: gpd.GeoDataFrame):
    """
    Calculates spatial similarity metrics between AI-detected and manual trackways.

    Args:
        ai_trackways_gdf (gpd.GeoDataFrame): GeoDataFrame of AI-detected trackways.
        manual_trackways_gdf (gpd.GeoDataFrame): GeoDataFrame of manual trackways.

    Returns:
        dict: A dictionary containing similarity metrics.
    """
    try:
        logger.info("Calculating similarity between AI and manual trackways.")

        # Ensure both GeoDataFrames have the same CRS
        if ai_trackways_gdf.crs != manual_trackways_gdf.crs:
            logger.warning("CRS mismatch. Reprojecting AI trackways to match manual trackways.")
            ai_trackways_gdf = ai_trackways_gdf.to_crs(manual_trackways_gdf.crs)

        # Spatial join to find intersecting trackways
        joined_gdf = gpd.sjoin(ai_trackways_gdf, manual_trackways_gdf, how="inner", op='intersects')

        if joined_gdf.empty:
            logger.warning("No intersecting trackways found.")
            return {
                "overlap_percentage": 0,
                "average_offset": None,
                "detection_completeness": 0,
                "matches": []
            }

        # Calculate overlap percentage
        total_manual_length = manual_trackways_gdf.geometry.length.sum()
        total_intersecting_length = 0

        # Group by manual trackway index to handle multiple intersections
        for index_right, group in joined_gdf.groupby('index_right'):
            manual_geom = manual_trackways_gdf.geometry.iloc[index_right]
            intersecting_geom = group.geometry.unary_union
            intersection = manual_geom.intersection(intersecting_geom)
            total_intersecting_length += intersection.length

        overlap_percentage = (total_intersecting_length / total_manual_length) * 100 if total_manual_length > 0 else 0

        # Calculate spatial offset (average Hausdorff distance)
        offsets = []
        for _, row in joined_gdf.iterrows():
            offset = row.geometry.hausdorff_distance(manual_trackways_gdf.geometry.iloc[row.index_right])
            offsets.append(offset)

        average_offset = sum(offsets) / len(offsets) if offsets else None

        # Calculate detection completeness
        detected_manual_trackways = joined_gdf['index_right'].nunique()
        detection_completeness = (detected_manual_trackways / len(manual_trackways_gdf)) * 100

        # Detailed matches
        matches = joined_gdf.apply(lambda row: {
            'ai_trackway_id': row.trackway_id,
            'manual_trackway_id': row.index_right,
            'offset': row.geometry.hausdorff_distance(manual_trackways_gdf.geometry.iloc[row.index_right])
        }, axis=1).tolist()

        metrics = {
            "overlap_percentage": overlap_percentage,
            "average_offset": average_offset,
            "detection_completeness": detection_completeness,
            "matches": matches
        }

        logger.info(f"Similarity calculation complete: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        raise e

import folium
import rasterio
from rasterio.transform import from_bounds
import numpy as np

def visualize_comparison(ai_trackways_gdf: gpd.GeoDataFrame, manual_trackways_gdf: gpd.GeoDataFrame, imagery_path: str = None, output_path: str = "comparison_map.html"):
    """
    Creates an interactive map to visualize the comparison between AI and manual trackways.

    Args:
        ai_trackways_gdf (gpd.GeoDataFrame): GeoDataFrame of AI-detected trackways.
        manual_trackways_gdf (gpd.GeoDataFrame): GeoDataFrame of manual trackways.
        imagery_path (str, optional): Path to the aerial imagery. Defaults to None.
        output_path (str, optional): Path to save the output HTML file. Defaults to "comparison_map.html".
    """
    try:
        logger.info(f"Creating visualization map at {output_path}")

        # Create a Folium map centered on the data
        center = manual_trackways_gdf.unary_union.centroid.coords[0][::-1]
        m = folium.Map(location=center, zoom_start=15)

        # Add aerial imagery if provided
        if imagery_path:
            with rasterio.open(imagery_path) as r:
                bounds = r.bounds
                image = r.read()
                # Transpose for folium
                image = np.transpose(image, (1, 2, 0))
                # Normalize if necessary (assuming 8-bit image)
                if image.max() > 1:
                    image = image / 255.0

                folium.raster_layers.ImageOverlay(
                    image=image,
                    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                    opacity=0.7,
                    name='Aerial Imagery'
                ).add_to(m)

        # Add manual trackways to the map
        folium.GeoJson(
            manual_trackways_gdf,
            name='Manual Trackways',
            style_function=lambda x: {'color': 'blue', 'weight': 3, 'opacity': 0.7}
        ).add_to(m)

        # Add AI trackways to the map
        folium.GeoJson(
            ai_trackways_gdf,
            name='AI-detected Trackways',
            style_function=lambda x: {'color': 'red', 'weight': 2, 'opacity': 0.8}
        ).add_to(m)

        folium.LayerControl().add_to(m)
        m.save(output_path)
        logger.info(f"Successfully saved map to {output_path}")

    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        raise e

import time

def generate_report(comparison_metrics: dict, ai_processing_time: float, manual_processing_time: float, output_path: str = "report.txt"):
    """
    Generates a report comparing AI vs. manual mapping approaches.

    Args:
        comparison_metrics (dict): The metrics from the similarity calculation.
        ai_processing_time (float): The time taken for AI processing in seconds.
        manual_processing_time (float): The time taken for manual processing in seconds.
        output_path (str, optional): The path to save the report. Defaults to "report.txt".
    """
    try:
        logger.info(f"Generating report at {output_path}")

        with open(output_path, "w") as f:
            f.write("AI vs. Manual Trackway Mapping Analysis Report\n")
            f.write("="*50 + "\n")
            f.write(f"Report generated on: {time.ctime()}\n\n")

            f.write("1. Accuracy Assessment\n")
            f.write("-" * 20 + "\n")
            f.write(f"Detection Completeness: {comparison_metrics.get('detection_completeness', 'N/A'):.2f}%\n")
            f.write(f"Overlap Percentage: {comparison_metrics.get('overlap_percentage', 'N/A'):.2f}%\n")
            f.write(f"Average Spatial Offset: {comparison_metrics.get('average_offset', 'N/A'):.2f} units\n\n")

            f.write("2. Time & Resource Requirement Analysis\n")
            f.write("-" * 20 + "\n")
            f.write(f"AI Processing Time: {ai_processing_time:.2f} seconds\n")
            f.write(f"Manual Processing Time: {manual_processing_time:.2f} seconds\n")

            time_saving = manual_processing_time - ai_processing_time
            f.write(f"Time Saving with AI: {time_saving:.2f} seconds\n")

            if ai_processing_time > 0:
                efficiency_gain = (manual_processing_time / ai_processing_time)
                f.write(f"Efficiency Gain: {efficiency_gain:.2f}x\n\n")

            f.write("3. Detailed Matches\n")
            f.write("-" * 20 + "\n")
            matches = comparison_metrics.get('matches', [])
            if matches:
                f.write(f"{'AI ID':<10} | {'Manual ID':<10} | {'Offset':<10}\n")
                f.write("-" * 34 + "\n")
                for match in matches:
                    f.write(f"{match['ai_trackway_id']:<10} | {match['manual_trackway_id']:<10} | {match['offset']:.2f}\n")
            else:
                f.write("No matches found.\n")

        logger.info(f"Successfully generated report at {output_path}")

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise e
