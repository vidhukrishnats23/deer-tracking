from pydantic_settings import BaseSettings
from typing import List, Tuple

class Settings(BaseSettings):
    app_name: str = "My FastAPI App"
    admin_email: str = "admin@example.com"

    # Ingestion settings
    allowed_mime_types: List[str] = ["image/jpeg", "image/png", "image/tiff"]
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    max_resolution: Tuple[int, int] = (4096, 4096)
    upload_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    labels_dir: str = "data/labels"
    metadata_log_file: str = "data/metadata.log"

    # Augmentation settings
    augmentation_rotation_angle: int = 15
    augmentation_scale_factor: float = 0.9
    augmentation_flip: bool = True
    normalized_size: Tuple[int, int] = (1024, 1024)

    # YOLO Training settings
    yolo_model: str = "yolov8n.pt"
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640
    project_name: str = "yolo_training"
    run_name: str = "exp"
    data_config: str = "data/data.yaml"

    class Config:
        env_file = ".env"

settings = Settings()
