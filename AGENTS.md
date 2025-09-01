# Agent Instructions

This document provides instructions for agents on how to work with this codebase.

## YOLO Training

This project includes a YOLO training pipeline. Here's how to use it.

### 1. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

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

### 3. Training Configuration

The training hyperparameters are managed in `app/config.py`. You can modify the `Settings` class to change the following parameters:

- `yolo_model`: The YOLO model to use (e.g., "yolov8n.pt").
- `epochs`: The number of training epochs.
- `batch_size`: The batch size for training.
- `img_size`: The image size for training.
- `project_name`: The name of the project where the models will be saved.
- `run_name`: The name of the specific training run.
- `data_config`: The path to the data configuration file.

### 4. Running the Training

To start the training, run the following command from the root of the project:

```bash
python scripts/train.py
```

The trained model and other artifacts will be saved in the `yolo_training/<run_name>` directory, as specified by the `project_name` and `run_name` in the configuration.
