import argparse
import json
import os

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import wasserstein_distance, wasserstein_distance_nd
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from datasets import UnlearnCifar10, UnlearnCifar100, UnlearnSVNH, get_dataloaders
from utils import load_checkpoint
from PIL import Image


def make_views(
    ax, angles, elevation=None, width=4, height=3, prefix="tmprot_", **kwargs
):
    """
    Makes jpeg pictures of the given 3d ax, with different angles.
    Args:
        ax (3D axis): te ax
        angles (list): the list of angles (in degree) under which to
                       take the picture.
        width,height (float): size, in inches, of the output images.
        prefix (str): prefix for the files created.

    Returns: the list of files created (for later removal)
    """

    files = []
    ax.figure.set_size_inches(width, height)

    for i, angle in enumerate(angles):
        ax.view_init(elev=elevation, azim=angle)
        fname = "%s%03d.jpeg" % (prefix, i)
        ax.figure.savefig(fname)
        files.append(fname)

    return files


def make_gif(files, output, delay=100, loop=0, **kwargs):
    """
    Create a GIF using Pillow.

    Args:
        :param files: List of image file paths to include in the GIF.
        :param output: Path to save the output GIF file.
        :param delay: Delay between frames in milliseconds.
        :param loop: Number of times the GIF should loop (0 means infinite).
    """
    # Open all images
    frames = [Image.open(file) for file in files]

    # Save the GIF
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],  # Include the other frames
        duration=delay,  # Duration of each frame in ms
        loop=loop,  # Number of loops
    )


def rotanimate(ax, angles, output, **kwargs):
    """
    Produces an animation (.gif) from a 3D plot on a 3D axis

    Args:
        ax (3D axis): the ax containing the plot of interest
        angles (list): the list of angles (in degree) under which to
                       show the plot.
        output : name of the output file. Must be .gif
        **kwargs:
            - width : in inches
            - heigth: in inches
            - delay : delay between frames in milliseconds
            - repeat : 0 for unlimited loop. Integer for n loops.
    """

    output_ext = os.path.splitext(output)[1]
    assert output_ext == ".gif", "The output file must be a .gif"
    files = make_views(ax, angles, **kwargs)

    make_gif(files, output, **kwargs)

    for f in files:
        os.remove(f)


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
        for folder in folders:

            data = json.load(open(os.path.join(folder, files[folder][cls])))

            features = np.array(data["all_features"][:SAMPLES])
            labels = np.array(data["all_labels"][:SAMPLES])

            class_separated = {}
            for idx, _ in enumerate(plot_classes):
                class_separated[idx] = features[labels == idx]
            all_features[folder] = class_separated

        # -----------------WASSERSTEIN DISTANCE ---------------------------
        print("Computing Wasserstein Distance")

        wass_mtxs = {}
        for folder in folders:

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
                    plt.imshow(delta, cmap="PiYG", interpolation="nearest")
                    plt.colorbar(label="Wasserstein Distance")
                    plt.title(
                        f"WD Delta Between classes in {folder1} and {folder2} exp {cls}"
                    )
                    plt.xlabel("Class")
                    plt.ylabel("Class")
                    plt.xticks(np.arange(len(plot_classes)), labels=plot_classes)
                    plt.yticks(np.arange(len(plot_classes)), labels=plot_classes)
                    plt.show()
