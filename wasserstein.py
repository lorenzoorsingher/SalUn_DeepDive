import argparse
import json
import os

import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import wasserstein_distance_nd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize")

    parser.add_argument(
        "--samples",
        "-S",
        type=int,
        default=200,
    )
    args = parser.parse_args()
    SAMPLES = args.samples

    plot_classes = [
        "Plane",
        "Car",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ]

    plot_colors = [
        "blue",
        "red",
        "green",
        "orange",
        "brown",
        "purple",
        "pink",
        "gray",
        "cyan",
        "yellow",
    ]

    folders = ["features/retrained", "features/salun_per_class", "features/base"]

    files = {
        folder: {int(n.split(".")[0].split("_")[-1]): n for n in os.listdir(folder)}
        for folder in folders
    }

    for cls, _ in enumerate(plot_classes):

        print(f"Current experiment: unlearned class {cls}")

        # -------------- DATA LOADING -----------------------------------
        all_features = {}
        all_labels_tsne = {}
        pre_tsne_feat = {}
        for folder in folders:

            data = json.load(open(os.path.join(folder, files[folder][cls])))

            features = np.array(data["all_features"][:SAMPLES])
            labels = np.array(data["all_labels"][:SAMPLES])

            class_separated = {}
            for idx, _ in enumerate(plot_classes):
                class_separated[idx] = features[labels == idx]
            all_features[folder] = class_separated

        # ----------------- WASSERSTEIN DISTANCE ---------------------------
        print("Computing Wasserstein Distance")

        wass_mtxs = {}
        for i, folder in enumerate(folders):

            wass_dist = np.zeros((len(plot_classes), len(plot_classes)))

            wass_dist[:] = np.nan
            class_separated = all_features[folder]

            for idx1, _ in enumerate(plot_classes):
                for idx2, _ in enumerate(plot_classes):

                    if idx1 < idx2:
                        distance = wasserstein_distance_nd(
                            class_separated[idx1], class_separated[idx2]
                        )
                        wass_dist[idx1, idx2] = distance
                        print(
                            f"Distance between {plot_classes[idx1]} and {plot_classes[idx2]} in {folder} is {distance}"
                        )

            wass_mtxs[folder] = wass_dist
            os.makedirs(f"images/{folder}", exist_ok=True)

            # Plot the heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(wass_dist, cmap="viridis", interpolation="nearest")
            plt.colorbar(label="Wasserstein Distance")
            plt.title(f"Wasserstein Distances Between Classes in {folder} exp {cls}")
            plt.xlabel("Class")
            plt.ylabel("Class")
            plt.xticks(np.arange(len(plot_classes)), labels=plot_classes)
            plt.yticks(np.arange(len(plot_classes)), labels=plot_classes)
            plt.show()

            # ----------------- COMPUTE DELTAS -------------------------------------------
            # breakpoint()
            for folder1 in folders:
                for folder2 in folders:
                    if folder1 != folder2:
                        delta = wass_mtxs[folder1] - wass_mtxs[folder2]
                        # breakpoint()
                        # Plot the heatmap
                        plt.figure(figsize=(10, 8))
                        plt.tight_layout()
                        fig = plt.figure()
                        plt.imshow(delta, cmap="PiYG", interpolation="nearest")
                        plt.colorbar(label="Wasserstein Distance")
                        plt.title(
                            f"WD Delta Between classes in {folder1} and {folder2} exp {cls}"
                        )
                        plt.xlabel("Class")
                        plt.ylabel("Class")
                        plt.xticks(np.arange(len(plot_classes)), labels=plot_classes)
                        plt.yticks(np.arange(len(plot_classes)), labels=plot_classes)
                        # plt.show()
                        plt.savefig(f"images/delta_{folder1}_{folder2}_class{cls}.png", dpi=fig.dpi)
