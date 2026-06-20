import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def dataset_file_checks(images_root, labels_root):
    print("\n--- 1. Dataset Integrity Checks ---")
    report = {}
    mismatches = {"images_without_label": [], "labels_without_image": []}

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(images_root, split)
        lbl_dir = os.path.join(labels_root, split)

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue

        img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(
            os.path.join(img_dir, "*.png")
        )
        img_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in img_paths}

        label_paths = glob.glob(os.path.join(lbl_dir, "*.txt"))
        label_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in label_paths}

        images_only = sorted(set(img_basenames) - set(label_basenames))
        labels_only = sorted(set(label_basenames) - set(img_basenames))

        report[split] = {
            "n_images": len(img_basenames),
            "n_labels": len(label_basenames),
            "images_without_label": len(images_only),
            "labels_without_image": len(labels_only),
        }

        mismatches["images_without_label"].extend([img_basenames[k] for k in images_only])
        mismatches["labels_without_image"].extend([label_basenames[k] for k in labels_only])

    print(
        f"{'Split':<10} | {'Images':<10} | {'Labels':<10} | {'Missing Labels':<15} | {'Missing Images':<15}"
    )
    print("-" * 75)
    for split, data in report.items():
        print(
            f"{split:<10} | {data['n_images']:<10} | {data['n_labels']:<10} | {data['images_without_label']:<15} | {data['labels_without_image']:<15}"
        )

    if mismatches["images_without_label"]:
        print(f"\nSample images without labels: {mismatches['images_without_label'][:5]}")
    if mismatches["labels_without_image"]:
        print(f"Sample labels without images: {mismatches['labels_without_image'][:5]}")

    return report, mismatches


def parse_yolo_label_file(label_path):
    items = []
    with open(label_path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Unexpected format in {label_path}:{lineno}: {line}")
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            items.append({"class": cls, "xc": xc, "yc": yc, "w": w, "h": h})
    return items


def validate_label_values(images_root, labels_root, min_box_area_ratio=0.0002):
    print("\n--- 2. Label Numeric Validation & Anomaly Detection ---")
    problems = []
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(images_root, split)
        lbl_dir = os.path.join(labels_root, split)

        if not os.path.exists(lbl_dir):
            continue

        for label_path in glob.glob(os.path.join(lbl_dir, "*.txt")):
            base = os.path.splitext(os.path.basename(label_path))[0]
            # Assumes jpg for simplicity, could be png
            image_path = os.path.join(img_dir, base + ".jpg")

            try:
                anns = parse_yolo_label_file(label_path)
            except Exception as e:
                problems.append(
                    {
                        "type": "parse_error",
                        "image": image_path,
                        "label": label_path,
                        "error": str(e),
                    }
                )
                continue

            for ann in anns:
                xc, yc, wb, hb = ann["xc"], ann["yc"], ann["w"], ann["h"]
                if not (
                    0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= wb <= 1.0 and 0.0 <= hb <= 1.0
                ):
                    problems.append(
                        {
                            "type": "out_of_range",
                            "image": image_path,
                            "label": label_path,
                            "ann": ann,
                        }
                    )
                if wb == 0 or hb == 0:
                    problems.append(
                        {"type": "zero_dim", "image": image_path, "label": label_path, "ann": ann}
                    )

                area_ratio = wb * hb
                if area_ratio < min_box_area_ratio:
                    problems.append(
                        {
                            "type": "tiny_box",
                            "image": image_path,
                            "label": label_path,
                            "ann": ann,
                            "area_ratio": area_ratio,
                        }
                    )

    print(f"Total problems found: {len(problems)}")
    problem_types = [p["type"] for p in problems]
    unique_types, counts = np.unique(problem_types, return_counts=True)
    for ptype, count in zip(unique_types, counts):
        print(f"  {ptype}: {count}")

    return problems


def dataset_bbox_eda(images_root, labels_root, output_dir="plots"):
    print("\n--- 3. Dataset-level EDA ---")
    os.makedirs(output_dir, exist_ok=True)
    area_ratios, widths, heights, aspect_ratios, center_xs, center_ys = [], [], [], [], [], []

    for split in ["train", "val", "test"]:
        lbl_dir = os.path.join(labels_root, split)

        if not os.path.exists(lbl_dir):
            continue

        for label_path in glob.glob(os.path.join(lbl_dir, "*.txt")):
            anns = parse_yolo_label_file(label_path)
            for ann in anns:
                xc, yc, wb, hb = ann["xc"], ann["yc"], ann["w"], ann["h"]
                area_ratios.append(wb * hb)
                widths.append(wb)
                heights.append(hb)
                aspect_ratios.append((wb) / (hb) if hb > 0 else np.nan)
                center_xs.append(xc)
                center_ys.append(yc)

    if not area_ratios:
        print("No bounding boxes found for EDA.")
        return

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(area_ratios, bins=50)
    plt.title("BBox Area (Normalized)")

    plt.subplot(1, 3, 2)
    plt.hist(widths, bins=50)
    plt.title("BBox Width (Normalized)")

    plt.subplot(1, 3, 3)
    plt.hist(heights, bins=50)
    plt.title("BBox Height (Normalized)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bbox_dims_histogram.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(5, 5))
    heatmap, xedges, yedges = np.histogram2d(center_xs, center_ys, bins=50, range=[[0, 1], [0, 1]])
    plt.imshow(heatmap.T[::-1], extent=[0, 1, 0, 1], origin="lower", aspect="auto")
    plt.title("BBox Center Heatmap (Normalized)")
    plt.xlabel("center_x")
    plt.ylabel("center_y")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "bbox_center_heatmap.png"), dpi=300)
    plt.close()

    print(f"Saved EDA plots to {output_dir}/")


def explore_dataset(dataset_path: str, num_samples: int = 5):
    """
    Run all 4 steps of Data Exploration:
    1. Dataset file checks
    2. Label validation
    3. Dataset-level EDA
    4. Visual sanity check
    """
    images_root = os.path.join(dataset_path, "images")
    labels_root = os.path.join(dataset_path, "labels")

    dataset_file_checks(images_root, labels_root)
    validate_label_values(images_root, labels_root)
    dataset_bbox_eda(images_root, labels_root)

    # 4. Visual sanity check (using existing logic, but refactored to look at train specifically)
    from src.utils.data import visualize_sample_data

    train_images = os.path.join(images_root, "train")
    train_labels = os.path.join(labels_root, "train")
    if os.path.exists(train_images) and os.path.exists(train_labels):
        print("\n--- 4. Visual Sanity Check ---")
        visualize_sample_data(train_images, train_labels, num_samples=num_samples)
    else:
        print("\nSkipping visual sanity check: Train split not found.")
