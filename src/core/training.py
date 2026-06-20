import os
import shutil
from typing import Tuple, Any

from ultralytics import YOLO

def train_model(
    data_yaml_path: str,
    model_weights: str = "derived/yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    workers: int = 4,
    img_size: int = 640,
    fraction: float = 1.0,
    patience: int = 10,
    output_weights_path: str = "models/best_license_plate_yolov8.pt",
) -> Tuple[YOLO, Any]:
    """
    Train a YOLO model for license plate detection.
    """
    print(f"Starting YOLO training for {epochs} epochs...")

    from ultralytics import settings

    settings.update({"weights_dir": os.path.abspath("derived")})

    # Initialize YOLO model
    model = YOLO(model_weights)

    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    project_dir = os.path.abspath("derived/training_runs")
    run_name = "license_plate_det"

    # Train
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=project_dir,
        name=run_name,
        save_period=2,
        fraction=fraction,
        patience=patience,
        workers=workers,
    )

    # Ultralytics internally downloads 'yolo26n.pt' to the cwd during its AMP checks.
    # We clean it up here so it doesn't clutter the project root.
    if os.path.exists("yolo26n.pt"):
        try:
            os.remove("yolo26n.pt")
        except Exception as e:
            print(f"Warning: Could not remove stray yolo26n.pt: {e}")

    # Copy best weights to models/ folder
    best_weights_src = os.path.join(project_dir, run_name, "weights", "best.pt")
    if os.path.exists(best_weights_src):
        os.makedirs(os.path.dirname(output_weights_path), exist_ok=True)
        shutil.copy(best_weights_src, output_weights_path)
        print(f"Successfully saved best weights to: {output_weights_path}")
    else:
        print(
            f"Warning: Best weights not found at {best_weights_src}. Did training complete successfully?"
        )

    # Generate plotting metrics
    from src.utils.plotting import plot_training_results

    results_csv = os.path.join(project_dir, run_name, "results.csv")
    plot_training_results(results_csv, "plots/training_metrics.png")

    print("Training complete.")
    return model, results
