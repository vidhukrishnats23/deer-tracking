import os
import sys
import base64
import time
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from app.config import settings

def generate_html_report(results, run_name):
    """
    Generate an HTML report from the training results.
    """
    save_dir = Path(results.save_dir)
    report_dir = Path('reports')
    report_dir.mkdir(exist_ok=True)

    # Read metrics from results.csv
    try:
        metrics_df = pd.read_csv(save_dir / 'results.csv')
        # Strip whitespace from column names
        metrics_df.columns = metrics_df.columns.str.strip()
        metrics_html = metrics_df.to_html(index=False)
    except FileNotFoundError:
        metrics_html = "<p>Could not find results.csv.</p>"

    # Embed images in HTML
    images_html = "<h2>Plots</h2>"
    image_files = list(save_dir.glob('*.png')) + list(save_dir.glob('*.jpg'))
    for img_path in image_files:
        try:
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            images_html += f'<h3>{img_path.name}</h3><img src="data:image/png;base64,{img_data}" alt="{img_path.name}" style="max-width:100%;"><br>'
        except Exception as e:
            images_html += f"<p>Could not embed image {img_path.name}: {e}</p>"

    # Get training logs
    log_path = save_dir / 'args.yaml'
    try:
        with open(log_path, 'r') as f:
            logs = f.read().replace('\n', '<br>')
        logs_html = f"<h2>Training Arguments</h2><p><pre>{logs}</pre></p>"
    except FileNotFoundError:
        logs_html = "<h2>Training Arguments</h2><p>Could not find args.yaml.</p>"


    html_content = f"""
    <html>
    <head>
        <title>YOLO Training Report - {run_name}</title>
        <style>
            body {{ font-family: sans-serif; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
            th {{ background-color: #f2f2f2; }}
            img {{ max-width: 800px; height: auto; }}
        </style>
    </head>
    <body>
        <h1>YOLO Training Report - {run_name}</h1>
        <h2>Metrics</h2>
        {metrics_html}
        {images_html}
        {logs_html}
    </body>
    </html>
    """

    report_path = report_dir / f'report_{run_name}.html'
    with open(report_path, 'w') as f:
        f.write(html_content)
    print(f"HTML report generated at {report_path}")


def train():
    """
    Train the YOLO model with the settings specified in the config file.
    """
    # Generate a unique run name based on timestamp
    run_name = time.strftime("%Y%m%d-%H%M%S")

    # Load the YOLO model
    model = YOLO(settings.yolo_model)

    # Train the model
    results = model.train(
        data=settings.data_config,
        epochs=settings.epochs,
        batch=settings.batch_size,
        imgsz=settings.img_size,
        project=settings.project_name,
        name=run_name,
        plots=True  # Enable plot generation
    )

    # The model is saved automatically by ultralytics in the project/name directory
    print(f"Model training complete. The model is saved in the '{results.save_dir}' directory.")

    # Generate HTML report
    generate_html_report(results, run_name)


if __name__ == "__main__":
    train()
