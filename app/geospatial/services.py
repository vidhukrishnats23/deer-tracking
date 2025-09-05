import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
import geopandas as gpd

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
