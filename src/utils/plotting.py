import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_training_results(csv_path: str, save_path: str):
    """
    Reads YOLO's results.csv and generates a 4x4 dynamic plot
    of training and validation loss and accuracy metrics vs epoch.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Cannot generate plot.")
        return

    try:
        # Read the CSV file and strip whitespace from column names
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Filter columns to only plot the 6 losses and mAP50-95
        cols_to_plot = [c for c in df.columns if ("loss" in c.lower() or "map50-95" in c.lower())]

        # We need a 2x4 subplot to fit 7 plots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        # Plot up to the available slots
        for i, col in enumerate(cols_to_plot):
            if i >= len(axes):
                break

            ax = axes[i]
            ax.plot(df["epoch"], df[col], marker="o", linestyle="-", linewidth=2, markersize=4)
            ax.set_title(col, fontsize=12)
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.7)

        # Hide any unused subplots
        for j in range(len(cols_to_plot), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Training metrics plot saved to {save_path}")

    except Exception as e:
        print(f"Failed to generate training plot: {e}")
