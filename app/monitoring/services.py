from app.trackways.services import analyze_trackways
from typing import Dict, Any, Optional
from app.logger import logger
import geopandas as gpd
from shapely.geometry import LineString
from app.config import settings
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import numpy as np
import os
import time
import folium
from scipy.stats import ttest_ind, t

def _trackways_to_gdf(trackways: dict) -> gpd.GeoDataFrame:
    """Converts a dictionary of trackways to a GeoDataFrame."""
    geometries = []
    if not trackways:
        return gpd.GeoDataFrame(geometries, crs=settings.TARGET_CRS)
    for trackway_id, data in trackways.items():
        points = data.get("points", [])
        if len(points) < 2: continue
        line = LineString([(p['x'], p['y']) for p in points])
        geometries.append({'geometry': line, 'properties': {'trackway_id': trackway_id, 'length': data.get('length'), 'avg_speed': data.get('average_speed')}})
    if not geometries:
        return gpd.GeoDataFrame(geometries, columns=['geometry', 'trackway_id', 'length', 'avg_speed'], crs=settings.TARGET_CRS)
    return gpd.GeoDataFrame.from_features(geometries, crs=settings.TARGET_CRS)

def _compare_trackways(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, buffer_distance: float = 10.0) -> Dict[str, list]:
    """Compares two GeoDataFrames of trackways to find new, modified, and abandoned trackways."""
    if gdf1.empty and gdf2.empty: return {"new": [], "abandoned": [], "modified": []}
    if gdf1.empty: return {"new": gdf2['trackway_id'].tolist(), "abandoned": [], "modified": []}
    if gdf2.empty: return {"new": [], "abandoned": gdf1['trackway_id'].tolist(), "modified": []}
    buffered_gdf1 = gdf1.copy()
    buffered_gdf1['geometry'] = gdf1.geometry.buffer(buffer_distance)
    join_gdf = gpd.sjoin(gdf2, buffered_gdf1, how='left', op='intersects')
    new_ids = join_gdf[join_gdf['index_right'].isna()]['trackway_id_left'].unique().tolist()
    modified_gdf = join_gdf[~join_gdf['index_right'].isna()]
    modified_p2_ids = modified_gdf['trackway_id_left'].unique().tolist()
    modified_p1_ids = modified_gdf['trackway_id_right'].unique().tolist()
    abandoned_ids = list(set(gdf1['trackway_id'].tolist()) - set(modified_p1_ids))
    return {"new": new_ids, "abandoned": abandoned_ids, "modified": modified_p2_ids}

def _generate_impact_intensity_map(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, cell_size: float) -> Optional[str]:
    """Generates an impact intensity change map from two sets of trackways."""
    if gdf1.empty and gdf2.empty: return None
    if gdf1.empty: total_bounds = gdf2.total_bounds
    elif gdf2.empty: total_bounds = gdf1.total_bounds
    else:
        minx1, miny1, maxx1, maxy1 = gdf1.total_bounds
        minx2, miny2, maxx2, maxy2 = gdf2.total_bounds
        total_bounds = [min(minx1, minx2), min(miny1, miny2), max(maxx1, maxx2), max(maxy1, maxy2)]

    minx, miny, maxx, maxy = total_bounds
    width, height = int(np.ceil((maxx - minx) / cell_size)), int(np.ceil((maxy - miny) / cell_size))
    if width == 0 or height == 0: return None
    transform = from_origin(minx, maxy, cell_size, cell_size)
    out_shape = (height, width)
    grid1 = rasterize(shapes=((g, 1) for g in gdf1.geometry), out_shape=out_shape, transform=transform, fill=0, all_touched=True, merge_alg=rasterio.enums.MergeAlg.add, dtype=rasterio.int16) if not gdf1.empty else np.zeros(out_shape, dtype=rasterio.int16)
    grid2 = rasterize(shapes=((g, 1) for g in gdf2.geometry), out_shape=out_shape, transform=transform, fill=0, all_touched=True, merge_alg=rasterio.enums.MergeAlg.add, dtype=rasterio.int16) if not gdf2.empty else np.zeros(out_shape, dtype=rasterio.int16)
    change_grid = grid2 - grid1
    output_path = os.path.join(settings.processed_dir, f"impact_change_map_{time.strftime('%Y%m%d-%H%M%S')}.tif")
    os.makedirs(settings.processed_dir, exist_ok=True)
    profile = {'driver': 'GTiff', 'height': height, 'width': width, 'count': 1, 'dtype': change_grid.dtype, 'crs': settings.TARGET_CRS, 'transform': transform, 'nodata': -9999}
    with rasterio.open(output_path, 'w', **profile) as dst: dst.write(change_grid, 1)
    logger.info(f"Impact intensity change map saved to {output_path}")
    return output_path

def _visualize_temporal_changes(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, change_summary: dict, output_dir: str = "reports") -> Optional[str]:
    """Creates an interactive map to visualize temporal changes in trackways."""
    if gdf1.empty and gdf2.empty: return None
    try:
        center_gdf = gdf2 if not gdf2.empty else gdf1
        if center_gdf.geometry.is_empty.all(): return None
        center = center_gdf.to_crs(epsg=4326).unary_union.centroid.coords[0][::-1]
        m = folium.Map(location=center, zoom_start=14)
        if (abandoned_ids := change_summary.get('abandoned', [])) and not gdf1.empty:
            if not (abandoned_gdf := gdf1[gdf1['trackway_id'].isin(abandoned_ids)]).empty: folium.GeoJson(abandoned_gdf.to_crs(epsg=4326), name='Abandoned', style_function=lambda x: {'color': 'blue', 'weight': 2}).add_to(m)
        if (new_ids := change_summary.get('new', [])) and not gdf2.empty:
            if not (new_gdf := gdf2[gdf2['trackway_id'].isin(new_ids)]).empty: folium.GeoJson(new_gdf.to_crs(epsg=4326), name='New', style_function=lambda x: {'color': 'red', 'weight': 3}).add_to(m)
        if (modified_ids := change_summary.get('modified', [])) and not gdf2.empty:
            if not (modified_gdf := gdf2[gdf2['trackway_id'].isin(modified_ids)]).empty: folium.GeoJson(modified_gdf.to_crs(epsg=4326), name='Modified/Existing', style_function=lambda x: {'color': 'green', 'weight': 2.5}).add_to(m)
        folium.LayerControl().add_to(m)
        os.makedirs(output_dir, exist_ok=True)
        map_path = os.path.join(output_dir, f"change_visualization_map_{time.strftime('%Y%m%d-%H%M%S')}.html")
        m.save(map_path)
        logger.info(f"Change visualization map saved to {map_path}")
        return map_path
    except Exception as e:
        logger.error(f"Error creating visualization map: {e}")
        return None

def _calculate_statistical_trends(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> dict:
    """Calculates statistical trends by comparing trackway metrics between two periods."""
    stats_results = {}
    for metric in ['length', 'avg_speed']:
        s1, s2 = gdf1[metric].dropna(), gdf2[metric].dropna()
        if len(s1) < 2 or len(s2) < 2:
            stats_results[metric] = {"p_value": None, "significant_change": False, "period1_mean": s1.mean() if not s1.empty else 0, "period2_mean": s2.mean() if not s2.empty else 0, "confidence_interval1": (None, None), "confidence_interval2": (None, None)}
            continue
        stat, p_value = ttest_ind(s1, s2, equal_var=False)
        ci1 = t.interval(0.95, len(s1)-1, loc=s1.mean(), scale=s1.sem()) if len(s1) > 1 else (None, None)
        ci2 = t.interval(0.95, len(s2)-1, loc=s2.mean(), scale=s2.sem()) if len(s2) > 1 else (None, None)
        stats_results[metric] = {"p_value": p_value, "significant_change": p_value < 0.05, "period1_mean": s1.mean(), "period2_mean": s2.mean(), "confidence_interval1": ci1, "confidence_interval2": ci2}
    return stats_results

def _generate_monitoring_report(analysis_results: dict, output_dir: str = "reports") -> Optional[str]:
    """Generates an HTML report from the temporal analysis results."""
    def format_stats(metric_name, stats):
        metric_stats = stats.get(metric_name, {})
        if not metric_stats or metric_stats.get('p_value') is None: return f"<tr><td colspan='2'>Not enough data for {metric_name} analysis.</td></tr>"
        ci1, ci2 = metric_stats['confidence_interval1'], metric_stats['confidence_interval2']
        return f"""<tr><td>Mean (Period 1)</td><td>{metric_stats['period1_mean']:.2f}</td></tr>
                   <tr><td>95% CI (Period 1)</td><td>({ci1[0]:.2f}, {ci1[1]:.2f})</td></tr>
                   <tr><td>Mean (Period 2)</td><td>{metric_stats['period2_mean']:.2f}</td></tr>
                   <tr><td>95% CI (Period 2)</td><td>({ci2[0]:.2f}, {ci2[1]:.2f})</td></tr>
                   <tr><td>P-value (t-test)</td><td>{metric_stats['p_value']:.4f}</td></tr>
                   <tr><td>Significant Change (p<0.05)</td><td>{'Yes' if metric_stats['significant_change'] else 'No'}</td></tr>"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp, report_path = time.strftime("%Y%m%d-%H%M%S"), os.path.join(output_dir, f"monitoring_report_{time.strftime('%Y%m%d-%H%M%S')}.html")
        summary, map_path, viz_map_path, stats = analysis_results.get("change_summary", {}), analysis_results.get("intensity_map_path"), analysis_results.get("visualization_map_path"), analysis_results.get("statistical_summary", {})
        html = f"""<html><head><title>Monitoring Report - {timestamp}</title><style>body{{font-family:sans-serif;margin:2em}}h1,h2,h3{{color:#333}}table{{border-collapse:collapse;width:50%}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background-color:#f2f2f2}}</style></head><body>
            <h1>Temporal Monitoring Report</h1><p>Generated: {time.ctime()}</p>
            <h2>Summary of Changes</h2><table><tr><th>Metric</th><th>Value</th></tr>
                <tr><td># Trackways Period 1</td><td>{len(analysis_results.get("period1_trackways", {}))}</td></tr>
                <tr><td># Trackways Period 2</td><td>{len(analysis_results.get("period2_trackways", {}))}</td></tr>
                <tr><td>New Trackways</td><td>{len(summary.get("new", []))}</td></tr>
                <tr><td>Abandoned Trackways</td><td>{len(summary.get("abandoned", []))}</td></tr>
                <tr><td>Modified/Existing Trackways</td><td>{len(summary.get("modified", []))}</td></tr>
            </table>
            <h2>Impact Intensity Map</h2><p>Shows change in trackway density. Path: <a href='file://{os.path.abspath(map_path)}'>{map_path if map_path else 'N/A'}</a></p>
            <h2>Change Visualization Map</h2><p>Interactive map of trackway changes. Path: <a href='file://{os.path.abspath(viz_map_path)}'>{viz_map_path if viz_map_path else 'N/A'}</a></p>
            <h2>Statistical Trend Analysis</h2><h3>Trackway Length</h3><table>{format_stats('length', stats)}</table><h3>Average Speed</h3><table>{format_stats('avg_speed', stats)}</table>
            </body></html>"""
        with open(report_path, "w") as f: f.write(html)
        logger.info(f"Monitoring report saved to {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Error generating monitoring report: {e}")
        return None

# --- Automated Scheduling Note ---
# The temporal_analysis function is designed to be called programmatically. To automate
# the periodic re-analysis of imagery datasets, this function can be triggered by an
# external scheduling system.
#
# Common approaches include:
# 1. Cron Job: A simple cron job on a Linux server could use a tool like `curl`
#    to make a POST request to the /monitoring/temporal_analysis endpoint at regular
#    intervals. See docs/scheduling.md for an example.
#
# 2. Task Queue (e.g., Celery with Celery Beat): For more complex scheduling needs,
#    integrating a task queue like Celery would be a robust solution. You could define a
#    periodic task that calls this service function directly. This is recommended for
#    production environments.
#
# Implementation of the scheduling system itself is considered an infrastructure-level
# task outside the scope of this application's codebase. See docs/scheduling.md for details.
# ---------------------------------
def temporal_analysis(
    start_date1: str, end_date1: str, start_date2: str, end_date2: str,
    buffer_distance: float = 10.0, cell_size: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """Performs temporal analysis by comparing trackways from two different time periods."""
    try:
        logger.info(f"Starting temporal analysis for {start_date1}-{end_date1} vs {start_date2}-{end_date2}.")
        trackways1, trackways2 = analyze_trackways(start_date=start_date1, end_date=end_date1) or {}, analyze_trackways(start_date=start_date2, end_date=end_date2) or {}
        gdf1, gdf2 = _trackways_to_gdf(trackways1), _trackways_to_gdf(trackways2)
        change_summary = _compare_trackways(gdf1, gdf2, buffer_distance)
        results = {
            "period1_trackways": trackways1, "period2_trackways": trackways2, "change_summary": change_summary,
            "intensity_map_path": _generate_impact_intensity_map(gdf1, gdf2, cell_size),
            "visualization_map_path": _visualize_temporal_changes(gdf1, gdf2, change_summary),
            "statistical_summary": _calculate_statistical_trends(gdf1, gdf2)
        }
        results["report_path"] = _generate_monitoring_report(results)
        return results
    except Exception as e:
        logger.error(f"An error occurred during temporal analysis: {e}")
        raise e
