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

To start the training, it is recommended to use the `Makefile`. This provides a simple way to run the entire pipeline.

From the root of the project, run:
```bash
make train
```

This command will execute the `scripts/train.py` script. The script now uses a timestamp-based unique ID for each run, so the artifacts for each training session will be saved in a new directory under `yolo_training/`. For example: `yolo_training/20250901-035239/`.

### 5. Evaluation Report

After the training is complete, an HTML report is automatically generated. This report contains the training metrics, plots (including loss curves and confusion matrix), and the training arguments.

The report is saved in the `reports/` directory, with a filename like `report_<run_name>.html`.
