import os
import glob
import random
from typing import List, Tuple

import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two bounding boxes (format: [x_center, y_center, width, height])."""
    # Convert to [x1, y1, x2, y2]
    b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2

    b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    # Intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_tp_fp_fn(preds: list, gts: list, iou_threshold: float = 0.5) -> Tuple[int, int, int]:
    """
    Computes True Positives, False Positives, and False Negatives for a single image
    using standard greedy matching based on IoU threshold.
    """
    tp, fp = 0, 0
    matched_gts = set()
    for p in preds:
        best_iou = 0
        best_gt_idx = -1
        for idx, g in enumerate(gts):
            if idx in matched_gts:
                continue
            iou = calculate_iou(g, p)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou > iou_threshold:
            tp += 1
            matched_gts.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gts) - len(matched_gts)
    return tp, fp, fn


def _calculate_custom_metrics(model: YOLO, data_yaml_path: str, conf_threshold: float) -> list:
    """Calculates custom evaluation metrics like MAE and Average IoU."""
    with open(data_yaml_path, "r") as f:
        data_config = yaml.safe_load(f)

    test_images_dir = os.path.join(data_config.get("path", ""), data_config.get("test", ""))
    test_labels_dir = test_images_dir.replace("images", "labels")

    if not os.path.exists(test_images_dir):
        print(f"Warning: Test images directory not found at {test_images_dir}. Skipping custom metrics.")
        return []

    print("Calculating custom metrics (MAE, Average IoU) on test dataset...")
    image_files = glob.glob(os.path.join(test_images_dir, "*.jpg")) + glob.glob(
        os.path.join(test_images_dir, "*.png")
    )

    ious = []
    maes = []
    tp, fp, fn = 0, 0, 0

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(test_labels_dir, label_name)

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        gt_boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, x_c, y_c, w_b, h_b = map(float, parts[:5])
                gt_boxes.append([x_c, y_c, w_b, h_b])

        if not gt_boxes:
            continue

        results = model.predict(img_path, verbose=False, conf=conf_threshold)
        pred_boxes = []
        for r in results:
            for box in r.boxes:
                xywhn = box.xywhn.cpu().numpy().squeeze()
                if len(xywhn.shape) == 0 or xywhn.size == 0:
                    continue
                if xywhn.size == 4:
                    pred_boxes.append(xywhn.tolist())

        if pred_boxes and gt_boxes:
            for gt_box in gt_boxes:
                best_iou = 0
                best_pred = None
                for pred_box in pred_boxes:
                    iou = calculate_iou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = pred_box

                ious.append(best_iou)
                if best_pred:
                    mae = np.mean(np.abs(np.array(gt_box) - np.array(best_pred)))
                    maes.append(mae)
                else:
                    ious.append(0.0)

        tp_img, fp_img, fn_img = compute_tp_fp_fn(pred_boxes, gt_boxes)
        tp += tp_img
        fp += fp_img
        fn += fn_img

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if (4 * precision + recall) > 0
        else 0.0
    )

    print("-" * 50)
    print("CUSTOM EVALUATION METRICS (Bounding Box)")
    print(f"Precision @ {conf_threshold:.2f}: {precision:.4f}")
    print(f"Recall    @ {conf_threshold:.2f}: {recall:.4f}")
    print(f"F2-Score  @ {conf_threshold:.2f}: {f2:.4f}")
    print(f"Average IoU: {np.mean(ious):.4f}" if ious else "Average IoU: N/A")
    print(
        f"Mean Absolute Error (normalized cx,cy,w,h): {np.mean(maes):.4f}"
        if maes
        else "MAE: N/A"
    )
    print("-" * 50)
    return image_files


def _run_yolo_validation(model: YOLO, data_yaml_path: str) -> None:
    """Runs standard YOLO PR-curve evaluation (mAP, F1-Max Precision/Recall)."""
    print("Running standard YOLO PR-curve evaluation (mAP, F1-Max Precision/Recall)...")
    val_results = model.val(
        data=data_yaml_path,
        split="test",
        project="derived/evaluation",
        name="test_metrics",
        exist_ok=True,
    )

    print("-" * 50)
    print("STANDARD EVALUATION METRICS")
    print(f"mAP50:     {val_results.box.map50:.4f}")
    print(f"mAP50-95:  {val_results.box.map:.4f}")
    print(f"Precision: {val_results.box.p.mean():.4f}")
    print(f"Recall:    {val_results.box.r.mean():.4f}")
    print("-" * 50)


def _visualize_evaluation_samples(model: YOLO, image_files: list, conf_threshold: float) -> None:
    """Generates visualizations for 5 random images in the dataset."""
    print("Generating evaluation sample visualizations...")
    random.seed(30)
    if not image_files:
        return

    sample_images = random.sample(image_files, min(5, len(image_files)))
    fig, axes = plt.subplots(len(sample_images), 3, figsize=(15, 5 * len(sample_images)))
    if len(sample_images) == 1:
        axes = [axes]

    # Get blur intensity from config
    blur_intensity = 51
    if os.path.exists("config/config.yaml"):
        with open("config/config.yaml", "r") as f:
            app_config = yaml.safe_load(f)
        blur_intensity = app_config.get("inference", {}).get("blur_intensity", 51)
    
    if blur_intensity % 2 == 0:
        blur_intensity += 1

    for idx, img_p in enumerate(sample_images):
        img = cv2.imread(img_p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_bbox = img.copy()
        img_blur = img.copy()

        results = model.predict(img_p, verbose=False, conf=conf_threshold)
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy.cpu().numpy().squeeze()
                conf = box.conf.cpu().numpy().squeeze().item()
                if xyxy.size == 4:
                    x1, y1, x2, y2 = xyxy.astype(int)

                    # 1. BBox + Conf
                    cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f"{conf:.2f}"
                    cv2.putText(
                        img_bbox,
                        label,
                        (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 255),
                        5,
                        cv2.LINE_AA  
                    )

                    # 2. Blur
                    roi = img_blur[y1:y2, x1:x2]
                    if roi.shape[0] > 0 and roi.shape[1] > 0:
                        blurred_roi = cv2.GaussianBlur(roi, (blur_intensity, blur_intensity), 0)
                        img_blur[y1:y2, x1:x2] = blurred_roi

        axes[idx][0].imshow(img)
        axes[idx][0].axis("off")
        axes[idx][0].set_title(f"Original: {os.path.basename(img_p)}", fontsize=10)

        axes[idx][1].imshow(img_bbox)
        axes[idx][1].axis("off")
        axes[idx][1].set_title(
            f"Bounding Boxes (conf \u2265 {conf_threshold:.2f})", fontsize=10
        )

        axes[idx][2].imshow(img_blur)
        axes[idx][2].axis("off")
        axes[idx][2].set_title("Blurred Output", fontsize=10)

    plt.tight_layout()
    save_path = "plots/evaluation_samples.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Evaluation samples visualization saved to {save_path}")


def evaluate_model(
    data_yaml_path: str,
    model_weights: str = "models/best_license_plate_yolov8.pt",
    conf_threshold: float = 0.25,
) -> None:
    """
    Evaluate the model on the test dataset.
    Calculates explicit MAE and Average IoU, runs standard YOLO validation, and visualizes samples.
    """
    if not os.path.exists(model_weights):
        print(f"Error: Model weights not found at {model_weights}")
        return

    print(f"Loading trained model from {model_weights}...")
    model = YOLO(model_weights)

    # 1. Custom evaluation for MAE and Average IoU
    image_files = _calculate_custom_metrics(model, data_yaml_path, conf_threshold)

    # 2. Standard YOLO Validation metrics (mAP, Precision, Recall)
    _run_yolo_validation(model, data_yaml_path)

    # 3. Visualizing 5 random images
    _visualize_evaluation_samples(model, image_files, conf_threshold)


def optimize_conf_threshold(
    data_yaml_path: str, model_weights: str = "models/best_license_plate_yolov8.pt"
) -> None:
    """
    Finds the optimal confidence threshold to maximize the F2-score (heavily favoring recall)
    to minimize False Negatives (missed license plates).
    Updates config.yaml with the best threshold found.
    """
    if not os.path.exists(model_weights):
        print(f"Error: Model weights not found at {model_weights}")
        return

    print(f"Loading trained model from {model_weights}...")
    model = YOLO(model_weights)

    with open(data_yaml_path, "r") as f:
        data_config = yaml.safe_load(f)

    test_images_dir = os.path.join(data_config.get("path", ""), data_config.get("test", ""))
    test_labels_dir = test_images_dir.replace("images", "labels")

    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory not found at {test_images_dir}.")
        return

    print("Running baseline inference to collect all potential predictions...")
    image_files = glob.glob(os.path.join(test_images_dir, "*.jpg")) + glob.glob(
        os.path.join(test_images_dir, "*.png")
    )

    ground_truths = {}
    baseline_predictions = {}

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(test_labels_dir, label_name)

        gts = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, x_c, y_c, w_b, h_b = map(float, parts[:5])
                        gts.append([x_c, y_c, w_b, h_b])
        ground_truths[img_name] = gts

        results = model.predict(img_path, verbose=False, conf=0.01)
        preds = []
        for r in results:
            for box in r.boxes:
                xywhn = box.xywhn.cpu().numpy().squeeze()
                conf = box.conf.cpu().numpy().squeeze().item()
                if xywhn.size == 4:
                    preds.append((xywhn.tolist(), conf))
        baseline_predictions[img_name] = preds

    print("\nSweeping confidence thresholds to optimize F2-Score...")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F2-Score':<10}")
    print("-" * 50)

    best_f2 = -1.0
    best_thresh = 0.05

    for t in np.arange(0.05, 1.0, 0.05):
        tp, fp, fn = 0, 0, 0

        for img_name, gts in ground_truths.items():
            all_preds = baseline_predictions.get(img_name, [])
            valid_preds = [p[0] for p in all_preds if p[1] >= t]

            tp_img, fp_img, fn_img = compute_tp_fp_fn(valid_preds, gts)
            tp += tp_img
            fp += fp_img
            fn += fn_img

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f2 = 0.0
        if (4 * precision + recall) > 0:
            f2 = 5 * precision * recall / (4 * precision + recall)

        print(f"{t:<10.2f} | {precision:<10.4f} | {recall:<10.4f} | {f2:<10.4f}")

        if f2 > best_f2:
            best_f2 = f2
            best_thresh = t

    print("-" * 50)
    print(f"Optimal Threshold (Highest F2-Score): {best_thresh:.2f}")

    config_path = "config/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        if "inference" not in config_data:
            config_data["inference"] = {}
        config_data["inference"]["conf_threshold"] = float(round(best_thresh, 2))

        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
        print(
            f"Successfully updated 'inference.conf_threshold' in {config_path} to {best_thresh:.2f}"
        )
    else:
        print(f"Warning: {config_path} not found. Could not save optimal threshold.")
