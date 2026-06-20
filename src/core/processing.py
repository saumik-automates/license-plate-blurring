import time
import cv2
import numpy as np


def extract_detections(results):
    """Extract bounding boxes and confidences from YOLO results."""
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                detections.append(
                    {"bbox": (int(x1), int(y1), int(x2), int(y2)), "confidence": confidence}
                )
    return detections


def draw_bounding_boxes(img: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes around detected license plates."""
    out_img = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        # Draw rectangle
        cv2.rectangle(out_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # Add label
        label = f"Conf: {conf:.2f}"
        cv2.putText(
            out_img, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
        )
    return out_img


def blur_license_plates(img: np.ndarray, detections: list, blur_intensity: int = 51) -> np.ndarray:
    """Apply Gaussian blur to detected license plate regions."""
    # Ensure blur intensity is odd
    if blur_intensity % 2 == 0:
        blur_intensity += 1

    blurred_img = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        # Prevent out of bounds
        y1, y2 = max(0, y1), min(img.shape[0], y2)
        x1, x2 = max(0, x1), min(img.shape[1], x2)

        # Skip invalid boxes
        if y2 <= y1 or x2 <= x1:
            continue

        # Extract region
        plate_region = blurred_img[y1:y2, x1:x2]

        # Apply Gaussian blur
        blurred_region = cv2.GaussianBlur(plate_region, (blur_intensity, blur_intensity), 0)

        # Replace original region
        blurred_img[y1:y2, x1:x2] = blurred_region

    return blurred_img


def process_frame(
    model_handler, img: np.ndarray, conf_threshold: float = 0.3, blur_intensity: int = 51
):
    """
    Process a single frame.
    Returns:
        box_img: Image with bounding boxes drawn
        blur_img: Image with plates blurred
        inference_time: Time taken in seconds for inference
    """
    # Run inference
    start_time = time.time()
    results = model_handler.predict(img, conf_threshold=conf_threshold)
    end_time = time.time()

    inference_time = end_time - start_time

    # Extract detections
    detections = extract_detections(results)

    # Generate outputs
    box_img = draw_bounding_boxes(img, detections)
    blur_img = blur_license_plates(img, detections, blur_intensity=blur_intensity)

    return box_img, blur_img, inference_time
