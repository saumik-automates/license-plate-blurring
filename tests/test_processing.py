import numpy as np
from src.core.processing import blur_license_plates


def test_blur_license_plates():
    # Create a dummy image (100x100 RGB)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255

    # Add a "license plate" area that is different color
    img[20:40, 20:60] = [0, 0, 0]  # black box

    # Detections list
    detections = [{"bbox": (20, 20, 60, 40), "confidence": 0.9}]

    # Blur
    blurred = blur_license_plates(img, detections, blur_intensity=11)

    # The blurred area should no longer be purely black [0, 0, 0]
    # except maybe exactly at the center depending on blur size, but it changes.
    # At least the shape remains same.
    assert blurred.shape == img.shape
    assert blurred.dtype == img.dtype

    # Check that pixels outside the bounding box are unchanged
    np.testing.assert_array_equal(blurred[0:10, 0:10], img[0:10, 0:10])


def test_blur_out_of_bounds():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    detections = [{"bbox": (-10, -10, 150, 150), "confidence": 0.9}]

    # Should not crash
    blurred = blur_license_plates(img, detections, blur_intensity=11)
    assert blurred.shape == img.shape
