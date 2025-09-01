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
    metadata_log_file: str = "data/metadata.log"


    class Config:
        env_file = ".env"

settings = Settings()
