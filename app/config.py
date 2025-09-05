from pydantic_settings import BaseSettings
from typing import List, Tuple

class Settings(BaseSettings):
    app_name: str = "My FastAPI App"
    admin_email: str = "admin@example.com"

    # Ingestion settings
    allowed_mime_types: List[str] = ["image/jpeg", "image/png", "image/tiff"]
    max_file_size: int = 100 * 1024 * 1024  # 100 MB
    max_resolution: Tuple[int, int] = (10000, 10000)
    upload_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    labels_dir: str = "data/labels"
    metadata_log_file: str = "data/metadata.log"

    # Quality assessment settings
    blur_threshold: float = 100.0
    underexposure_threshold: float = 0.1
    overexposure_threshold: float = 0.1
    sensor_width_mm: float = 23.5  # Example for a crop-sensor camera
    max_gsd: float = 0.5  # meters/pixel

    # Retention policies
    processed_images_retention_days: int = 30
    training_artifacts_retention_days: int = 90

    # Augmentation settings
    augmentation_rotation_angle: int = 15
    augmentation_scale_factor: float = 0.9
    augmentation_flip: bool = True
    normalized_size: Tuple[int, int] = (1024, 1024)

    # Geospatial settings
    TARGET_CRS: str = "EPSG:4326"  # Default to WGS84
    DEFAULT_CRS: str = "EPSG:3857" # Default to Web Mercator
    GIS_DATA_PATH: str = "data/gis"
    DEM_PATH: str = "data/gis/dem.tif"  # Path to the Digital Elevation Model
    APPLY_ORTHO_ON_INGEST: bool = False  # Whether to apply orthorectification on ingest

    # YOLO Training settings
    yolo_model: str = "yolov8n.pt"
    yolo_model_path: str = "yolov8n.pt"
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640
    project_name: str = "yolo_training"
    run_name: str = "exp"
    data_config: str = "data/data.yaml"

    # Habitat Classification settings
    habitat_map_path: str = "data/gis/habitat_map.tif"
    degradation_map_path: str = "data/gis/degradation_map.tif"
    habitat_model: str = "yolov8n-cls.pt"
    habitat_model_path: str = "yolov8n-cls.pt"
    habitat_epochs: int = 50
    habitat_batch_size: int = 32
    habitat_img_size: int = 224
    habitat_project_name: str = "habitat_training"
    habitat_run_name: str = "habitat_exp"
    habitat_data_config: str = "data/habitat_data.yaml"

    class Config:
        env_file = ".env"

settings = Settings()
