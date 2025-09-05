import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling, Resampling
from rasterio.merge import merge
from shapely.geometry import box
import geopandas as gpd
from typing import List
import os

from app.config import settings

def reproject_image(input_path: str, output_path: str, target_crs: str = settings.TARGET_CRS):
    """
    Reprojects a raster image to a different CRS.
    """
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest)

def create_geodataframe(bounds, crs):
    """
    Creates a GeoDataFrame from a bounding box.
    """
    gseries = gpd.GeoSeries([box(*bounds)], crs=crs)
    gdf = gpd.GeoDataFrame(geometry=gseries)
    return gdf


def orthorectify_image(input_path: str, output_path: str, dem_path: str = settings.DEM_PATH):
    """
    Orthorectifies a raster image using a DEM.
    This is a simplified implementation. For best results, GDAL with RPCs is recommended.
    """
    with rasterio.open(input_path) as src:
        # This is a simplified approach. A real implementation would likely use
        # src.rpcs and gdal.Warp for orthorectification.
        # For now, we reproject with the same CRS, which can correct for some distortions
        # if the original image was not north-up.
        transform, width, height = calculate_default_transform(
            src.crs, src.crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': src.crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.cubic)

def mosaic_images(image_paths: List[str], output_path: str):
    """
    Mosaics multiple raster images into a single seamless dataset.
    """
    src_files_to_mosaic = []
    for fp in image_paths:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)

    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
    })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files_to_mosaic:
        src.close()
