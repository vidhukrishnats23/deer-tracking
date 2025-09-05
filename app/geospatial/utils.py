import rasterio
from pyproj import Proj, transform
import tempfile
import os
from fastapi import UploadFile
import exifread

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

def extract_spatial_metadata(image_path: str):
    """
    Extracts spatial metadata from a GeoTIFF file.
    """
    try:
        with rasterio.open(image_path) as dataset:
            metadata = {
                "crs": dataset.crs.to_string() if dataset.crs else None,
                "bounds": list(dataset.bounds),
                "transform": list(dataset.transform),
                "width": dataset.width,
                "height": dataset.height,
            }
            return metadata
    except Exception as e:
        return {"error": f"Could not extract spatial metadata: {e}"}


def extract_exif_metadata(file: UploadFile):
    """
    Extracts EXIF metadata from an image file.
    """
    file.file.seek(0)
    try:
        tags = exifread.process_file(file.file, details=False)
        exif_data = {}
        for tag, value in tags.items():
            if tag not in ('JPEGThumbnail', 'TIFFThumbnail'):
                exif_data[tag] = str(value)
        return exif_data
    except Exception as e:
        return {"error": f"Could not extract EXIF metadata: {e}"}
    finally:
        file.file.seek(0)


def extract_detailed_metadata(file: UploadFile):
    """
    Extracts detailed metadata (spatial and EXIF) from an image file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tiff") as tmp:
        file.file.seek(0)
        tmp.write(file.file.read())
        tmp_path = tmp.name
    file.file.seek(0)

    try:
        spatial_metadata = extract_spatial_metadata(tmp_path)
        exif_metadata = extract_exif_metadata(file)

        return {
            "spatial": spatial_metadata,
            "exif": exif_metadata
        }
    except Exception as e:
        return {"error": f"Could not extract detailed metadata: {e}"}
    finally:
        os.remove(tmp_path)

def pixel_to_geo(pixel_x, pixel_y, geotiff_path):
    """
    Convert a single pixel coordinate (x, y) to a geographic coordinate.
    """
    with rasterio.open(geotiff_path) as src:
        # The transform method maps pixel coordinates to geographic coordinates
        geo_x, geo_y = src.transform * (pixel_x, pixel_y)
        return geo_x, geo_y
