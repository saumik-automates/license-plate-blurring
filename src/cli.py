import argparse
import os
import yaml
from ultralytics import settings

# Redirect all YOLO downloads and runs to derived/ directory
settings.update(
    {"weights_dir": os.path.abspath("derived"), "runs_dir": os.path.abspath("derived/runs")}
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_explore_data(num_samples: int = 5) -> None:
    from src.utils.data import download_dataset
    from src.utils.exploration import explore_dataset

    config = load_config()
    dataset_name = config["training"]["dataset"]

    path = download_dataset(dataset_name)
    explore_dataset(path, num_samples=num_samples)


def run_train() -> None:
    from src.utils.data import download_dataset, create_dataset_yaml
    from src.core.training import train_model

    config = load_config()

    dataset_name = config["training"]["dataset"]
    path = download_dataset(dataset_name)

    yaml_path = create_dataset_yaml(path, "derived/dataset.yaml")

    trained_weights_dest = config["model"].get(
        "trained_weights", "models/best_license_plate_yolov8.pt"
    )

    train_model(
        data_yaml_path=yaml_path,
        model_weights=config["model"]["default_weights"],
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        workers=config["training"].get("workers", 4),
        img_size=config["model"]["img_size"],
        fraction=config["training"].get("fraction", 1.0),
        patience=config["training"].get("patience", 10),
        output_weights_path=trained_weights_dest,
    )


def run_evaluate() -> None:
    from src.utils.data import download_dataset, create_dataset_yaml
    from src.core.evaluation import evaluate_model

    config = load_config()

    dataset_name = config["training"]["dataset"]
    path = download_dataset(dataset_name)
    yaml_path = create_dataset_yaml(path, "derived/dataset.yaml")

    trained_weights = config["model"].get("trained_weights", "models/best_license_plate_yolov8.pt")
    conf_threshold = config["inference"].get("conf_threshold", 0.3)

    evaluate_model(
        data_yaml_path=yaml_path, model_weights=trained_weights, conf_threshold=conf_threshold
    )


def run_optimize() -> None:
    from src.utils.data import download_dataset, create_dataset_yaml
    from src.core.evaluation import optimize_conf_threshold

    config = load_config()

    dataset_name = config["training"]["dataset"]
    path = download_dataset(dataset_name)
    yaml_path = create_dataset_yaml(path, "derived/dataset.yaml")

    trained_weights = config["model"].get("trained_weights", "models/best_license_plate_yolov8.pt")
    optimize_conf_threshold(data_yaml_path=yaml_path, model_weights=trained_weights)


def run_ui() -> None:
    # Import gradio app locally to avoid loading it if not needed
    from app.ui import demo

    print("Launching Gradio UI...")
    demo.launch(server_name="0.0.0.0", server_port=7860)


def run_api() -> None:
    import uvicorn
    from app.api import app

    config = load_config()
    host = config.get("api", {}).get("host", "0.0.0.0")
    port = config.get("api", {}).get("port", 8000)
    print(f"Launching FastAPI on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="License Plate Blurring System CLI")
    parser.add_argument(
        "--explore-data",
        type=int,
        nargs="?",
        const=16,
        help="Download dataset and perform data exploration on N samples (default: 16)",
    )
    parser.add_argument("--train", action="store_true", help="Train the YOLO model")
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate the YOLO model on test data"
    )
    parser.add_argument(
        "--optimize-threshold",
        action="store_true",
        help="Find the optimal confidence threshold to maximize F2-score",
    )
    parser.add_argument("--ui", action="store_true", help="Launch the Gradio Web UI")
    parser.add_argument("--api", action="store_true", help="Launch the FastAPI server")

    args = parser.parse_args()

    if args.explore_data is not None:
        run_explore_data(args.explore_data)
    elif args.train:
        run_train()
    elif args.evaluate:
        run_evaluate()
    elif args.optimize_threshold:
        run_optimize()
    elif args.ui:
        run_ui()
    elif args.api:
        run_api()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
