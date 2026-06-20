import os
import glob
import random
import pytest
import numpy as np
import cv2
import yaml

# Try to import FastAPI TestClient
try:
    from fastapi.testclient import TestClient
    from app.api import app

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from src.utils.data import download_dataset
from src.core.evaluation import compute_tp_fp_fn
from ultralytics import YOLO


@pytest.fixture(scope="module")
def dataset_paths():
    """Download the dataset dynamically and yield standard paths."""
    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    dataset_name = config.get("training", {}).get(
        "dataset", "fareselmenshawii/large-license-plate-dataset"
    )

    # Download dataset via kagglehub (extremely fast if already cached)
    base_path = download_dataset(dataset_name)

    # Dataset splits
    train_dir = os.path.join(base_path, "images", "train")
    val_dir = os.path.join(base_path, "images", "val")
    test_dir = os.path.join(base_path, "images", "test")

    return base_path, train_dir, val_dir, test_dir


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI or TestClient not installed")
def test_api_blurriness(dataset_paths):
    """
    Use the API Gateway to test actual images from the test folder.
    Check for blurriness (or at least valid image processing endpoint behavior).
    """
    client = TestClient(app)

    base_path, _, _, test_dir = dataset_paths

    image_files = glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(
        os.path.join(test_dir, "*.png")
    )
    if not image_files:
        pytest.skip("No test images found")

    # random.seed(42)
    sample_images = random.sample(image_files, min(5, len(image_files)))

    # Load config defaults
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    conf_threshold = config.get("inference", {}).get("conf_threshold", 0.3)
    blur_intensity = config.get("inference", {}).get("blur_intensity", 51)

    for img_path in sample_images:
        with open(img_path, "rb") as f:
            file_bytes = f.read()

        # 1. Send test image via TestClient
        response = client.post(
            "/predict/blur",
            files={"file": ("image.jpg", file_bytes, "image/jpeg")},
            data={"conf_threshold": conf_threshold, "blur_intensity": blur_intensity},
        )

        # 2. Check HTTP assertions
        assert response.status_code == 200, f"API returned status {response.status_code}"
        assert response.headers["content-type"] == "image/jpeg"

        # 3. Read the returned image
        nparr = np.frombuffer(response.content, np.uint8)
        blurred_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        orig_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        assert blurred_img.shape == orig_img.shape, "Blurred image shape mismatch"
        assert blurred_img is not None, "API returned empty or invalid image byte stream"


def test_model_precision_recall_lock(dataset_paths):
    """
    Randomly select hundreds of images from train, dev, and test.
    Run evaluation, calculate precision and recall.
    Lock down threshold values: Recall >= 0.85, Precision >= 0.80.
    """
    base_path, train_dir, val_dir, test_dir = dataset_paths

    # 1. Collect and sample
    all_images = []
    for d in [train_dir, val_dir, test_dir]:
        if os.path.exists(d):
            all_images.extend(glob.glob(os.path.join(d, "*.jpg")))
            all_images.extend(glob.glob(os.path.join(d, "*.png")))

    if not all_images:
        pytest.skip("No images found in dataset")

    num_samples = min(300, len(all_images))
    sample_images = random.sample(all_images, num_samples)

    # 2. Load the system state
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    trained_weights = config.get("model", {}).get(
        "trained_weights", "models/best_license_plate_yolov8.pt"
    )
    conf_threshold = config.get("inference", {}).get("conf_threshold", 0.5)

    if not os.path.exists(trained_weights):
        pytest.skip(f"Model weights not found: {trained_weights}")

    model = YOLO(trained_weights)

    tp, fp, fn = 0, 0, 0

    # 3. Compute Metrics Manually over sampled subset
    for img_path in sample_images:
        label_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"

        gts = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, x_c, y_c, w_b, h_b = map(float, parts[:5])
                        gts.append([x_c, y_c, w_b, h_b])

        results = model.predict(img_path, verbose=False, conf=conf_threshold)
        preds = []
        for r in results:
            for box in r.boxes:
                xywhn = box.xywhn.cpu().numpy().squeeze()
                if xywhn.size == 4:
                    preds.append(xywhn.tolist())

        tp_img, fp_img, fn_img = compute_tp_fp_fn(preds, gts)
        tp += tp_img
        fp += fp_img
        fn += fn_img

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print("\n--- EVALUATION LOCK METRICS ---")
    print(f"Sampled Images: {num_samples}")
    print(f"Precision: {precision:.4f} (Target >= 0.80)")
    print(f"Recall:    {recall:.4f} (Target >= 0.85)")
    print("-------------------------------")

    # 4. Strict assertions
    assert recall >= 0.85, (
        f"Recall dropped below threshold! (Current: {recall:.4f}, Target: >= 0.85)"
    )
    assert precision >= 0.80, (
        f"Precision dropped below threshold! (Current: {precision:.4f}, Target: >= 0.80)"
    )
