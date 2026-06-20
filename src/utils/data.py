import os
import glob
import yaml
import kagglehub
import cv2
import matplotlib.pyplot as plt
import random


def download_dataset(dataset_name: str = "fareselmenshawii/large-license-plate-dataset") -> str:
    """Download dataset using kagglehub."""
    print(f"Downloading dataset '{dataset_name}'...")
    path = kagglehub.dataset_download(dataset_name)
    print(f"Dataset downloaded to {path}")
    return path


def create_dataset_yaml(base_path: str, output_yaml: str = "derived/dataset.yaml") -> str:
    """Create YOLO format dataset.yaml."""
    os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
    yaml_content = {
        "path": base_path,
        "train": os.path.join("images", "train"),
        "val": os.path.join("images", "val"),
        "test": os.path.join("images", "test"),
        "nc": 1,
        "names": ["license_plate"],
    }

    with open(output_yaml, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"Created configuration file at {output_yaml}")
    return output_yaml


def visualize_sample_data(images_path: str, labels_path: str, num_samples: int = 5):
    """Visualize a few samples from the dataset in a single figure with subplots."""
    image_files = glob.glob(os.path.join(images_path, "*.jpg")) + glob.glob(
        os.path.join(images_path, "*.png")
    )

    if not image_files:
        print(f"No images found in {images_path}")
        return

    actual_samples = min(num_samples, len(image_files))
    print(f"Visualizing {actual_samples} samples...")

    # Calculate grid size (up to 3 columns)
    cols = min(3, actual_samples)
    rows = (actual_samples + cols - 1) // cols if cols > 0 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

    # Standardize axes to a list
    if actual_samples == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        # axes is 1D array
        axes = list(axes)
    else:
        # axes is 2D array
        axes = list(axes.flatten())

    # Randomly select samples with a fixed seed for reproducibility
    random.seed(20)
    selected_files = random.sample(image_files, actual_samples)

    for i, img_path in enumerate(selected_files):
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_path, label_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO format: class x_center y_center width height
                    _, x_c, y_c, w_b, h_b = map(float, parts[:5])

                    x1 = int((x_c - w_b / 2) * w)
                    y1 = int((y_c - h_b / 2) * h)
                    x2 = int((x_c + w_b / 2) * w)
                    y2 = int((y_c + h_b / 2) * h)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        ax = axes[i]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(img_name)
        ax.axis("off")

    # Turn off axes for any remaining unused subplots
    for j in range(actual_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    save_path = "plots/dataset_visualization.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Dataset visualization saved to {save_path}")
    plt.close(fig)
