import rasterio
from pyproj import Proj, transform
import tempfile
import os
from fastapi import UploadFile

def get_image_bounds(image_path: str):
    """
    Extracts the geographic bounds of a raster image.
    """
    with rasterio.open(image_path) as src:
        return src.bounds

def convert_coordinates(x: float, y: float, src_crs: str, dst_crs: str):
    """
    Converts coordinates from a source CRS to a destination CRS.
    """
    in_proj = Proj(f"epsg:{src_crs.split(':')[-1]}")
    out_proj = Proj(f"epsg:{dst_crs.split(':')[-1]}")
    x2, y2 = transform(in_proj, out_proj, x, y, always_xy=True)
    return x2, y2

def extract_spatial_metadata(file: UploadFile):
    """
    Extracts spatial metadata from a GeoTIFF file by saving it to a temporary file first.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tiff") as tmp:
        file.file.seek(0)
        tmp.write(file.file.read())
        tmp_path = tmp.name
    file.file.seek(0)  # Reset file pointer for subsequent reads

    try:
        with rasterio.open(tmp_path) as dataset:
            metadata = {
                "crs": dataset.crs.to_string() if dataset.crs else None,
                "bounds": dataset.bounds,
                "transform": dataset.transform,
                "width": dataset.width,
                "height": dataset.height,
            }
            return metadata
    except Exception as e:
        return {"error": f"Could not extract spatial metadata: {e}"}
    finally:
        os.remove(tmp_path)
