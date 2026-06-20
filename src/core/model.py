# src/core/model.py
import os
from ultralytics import YOLO


class ModelHandler:
    """Handles YOLOv8 model loading, downloading weights, and running inference."""

    def __init__(self, trained_path: str):
        self.trained_path = trained_path
        self.model = None

    def load_model(self) -> YOLO:
        """Loads the YOLO model. Throws an error if trained model does not exist."""
        if not self.trained_path or not os.path.exists(self.trained_path):
            raise FileNotFoundError(
                f"Required trained model weights were not found at '{self.trained_path}'. "
                "You must train the model first by running `python -m src.cli --train` before starting inference."
            )

        print(f"Loading custom trained model from {self.trained_path}...")
        self.model = YOLO(self.trained_path)
        return self.model

    def predict(self, img, conf_threshold: float = 0.3):
        """Runs inference on a single image or frame."""
        if self.model is None:
            self.load_model()

        results = self.model(img, conf=conf_threshold, verbose=False)
        return results
