# YOLOv8 FastAPI Application

## Overview

This project is a FastAPI application that provides a comprehensive toolkit for working with YOLOv8 models. It includes a robust API for image ingestion, annotation, and real-time prediction. The application is designed to be highly configurable and easy to extend, making it suitable for a wide range of computer vision tasks.

In addition to the API, this project also includes a complete pipeline for training YOLOv8 models on your own custom datasets. The training process is managed through a simple `Makefile` command and is highly customizable through configuration files.

## Features

- **Image Ingestion**: Upload images to the application for processing and prediction.
- **Image Annotation**: Create and manage annotations for your images.
- **Real-time Prediction**: Get real-time predictions from a trained YOLOv8 model.
- **YOLOv8 Training**: Train your own YOLOv8 models with a simple and efficient pipeline.
- **Configurable**: Easily configure the application and training pipeline to suit your needs.
- **Dockerized**: Run the application in a Docker container for easy deployment.

## API Endpoints

The application provides the following API endpoints:

- `POST /api/v1/ingest`: Upload an image for ingestion.
- `POST /api/v1/annotate`: Add an annotation to an image.
- `POST /api/v1/predict`: Get a prediction for an image.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You will need the following software to run this project:

- Python 3.9+
- Docker (optional)

### Installation

#### Local Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

#### Docker Installation

If you prefer to use Docker, you can build and run the application with `docker-compose`:

```bash
docker-compose up --build
```

This will build the Docker image and start the application. The API will be available at `http://localhost:8000`.

### Configuration

The application is configured through a combination of environment variables and a settings file.

1.  **Environment Variables**: Create a `.env` file in the root of the project by copying the `.env.example` file:

    ```bash
    cp .env.example .env
    ```

    You can then edit the `.env` file to set the `ADMIN_EMAIL` and other environment variables.

2.  **Settings File**: The main configuration file is `app/config.py`. You can modify this file to change the application's behavior, such as allowed MIME types, file sizes, and more. Any setting in this file can be overridden by setting an environment variable with the same name.

### Running the Application

-   **Locally**:

    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```

-   **With Docker**:

    ```bash
    docker-compose up
    ```

## YOLO Training

This project includes a pipeline for training YOLOv8 models.

### Dataset Setup

The training script expects the dataset to be in a specific format. The `data/data.yaml` file points to the training and validation data. By default, it's configured to use `data/train/images` and `data/valid/images`.

Make sure your dataset is structured as follows:

```
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       └── ...
└── valid/
    ├── images/
    │   ├── image2.jpg
    │   └── ...
    └── labels/
        ├── image2.txt
        └── ...
```

The `data/data.yaml` file should be updated to reflect the number of classes and their names.

### Training Configuration

The training hyperparameters are managed in `app/config.py`. You can modify the `Settings` class to change parameters such as the YOLO model, number of epochs, batch size, and more.

### Running the Training

To start the training, use the `Makefile`:

```bash
make train
```

This command will execute the `scripts/train.py` script. The training artifacts for each run will be saved in a new directory under `yolo_training/`.

### Evaluation Report

After training is complete, an HTML report is automatically generated in the `reports/` directory. This report contains training metrics, plots, and other useful information.

## Testing

To run the test suite, use `pytest`:

```bash
pytest
```

## Project Structure

```
.
├── app/                  # Main application code
│   ├── annotation/       # Annotation-related code
│   ├── ingestion/        # Ingestion-related code
│   ├── prediction/       # Prediction-related code
│   ├── processing/       # Image processing code
│   ├── __init__.py
│   ├── config.py         # Application configuration
│   └── main.py           # FastAPI application entrypoint
├── data/                 # Data for training and processing
├── scripts/              # Training scripts
│   └── train.py
├── tests/                # Test suite
├── .env.example          # Example environment file
├── .gitignore
├── AGENTS.md             # Instructions for agents
├── Dockerfile            # Dockerfile for the application
├── Dockerfile.gpu        # Dockerfile for GPU usage
├── Makefile              # Makefile for common commands
├── docker-compose.yml    # Docker Compose file
└── requirements.txt      # Python dependencies
```
