import argparse
import os

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import wasserstein_distance
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
        "--checkpoint",
        "-C",
        type=str,
        default="checkpoints/resnet18_cifar10_pretrained_forget.pt",
    )
    args = parser.parse_args()

    CHECKPOINT = args.checkpoint

    split = [0.7, 0.2, 0.1]
    model, config, transform, opt = load_checkpoint(CHECKPOINT)
    DSET = config["dataset"]

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

    (
        train_loader,
        val_loader,
        test_loader,
        forget_loader,
        retain_loader,
        _,
    ) = get_dataloaders(
        DSET,
        transform,
        unlr=0,
        itf=None,
        cf=None,
        batch_s=1,
        num_workers=2,
        pin_memory=False,
    )

    experiment = CHECKPOINT.split("/")[-1].split(".")[0]
    # Load pretrained model
    model.eval()

    # Hook to extract features from an intermediate layer
    features = []

    def hook(module, input, output):
        features.append(output)

    # Register the hook to a layer (e.g., avgpool layer)
    layer = model.global_pool
    layer.register_forward_hook(hook)

    # Extract latent features
    all_features = []
    all_labels = []

    with torch.no_grad():
        for idx, data in enumerate(tqdm(train_loader)):
            model(data["image"])
            latent_features = (
                features.pop().squeeze().view(data["image"].size(0), -1)
            )  # Flatten
            all_features.append(latent_features)
            all_labels.append(data["label"])

            if idx == 10000:
                break

    all_features = torch.cat(all_features).numpy()
    all_labels = torch.cat(all_labels).numpy()

    classes = np.unique(all_labels)
    centroids = {}
    for cls in classes:
        # Get features corresponding to the current class
        class_features = all_features[all_labels == cls]

        # Compute the mean (centroid) of the features for this class
        centroids[cls] = np.mean(class_features, axis=0)

    wasserstein_distances = {}
    for class1 in centroids:
        for class2 in centroids:
            if class1 < class2:
                distance = wasserstein_distance(centroids[class1], centroids[class2])
                wasserstein_distances[(class1, class2)] = distance

    num_classes = len(plot_classes)
    distance_matrix = np.zeros((num_classes, num_classes))
    for (class1, class2), distance in wasserstein_distances.items():
        distance_matrix[class1, class2] = distance
        distance_matrix[class2, class1] = distance  # Symmetric

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Wasserstein Distance")
    plt.title(
        f"Wasserstein Distances Between CIFAR-10 Class Centroids exp {experiment}"
    )
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.xticks(np.arange(num_classes), labels=plot_classes)
    plt.yticks(np.arange(num_classes), labels=plot_classes)

    # Add annotations for each cell
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:  # Skip diagonal
                plt.text(
                    j,
                    i,
                    f"{distance_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.show()

    # TSNE for 3D Visualization
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000)
    tsne_features_3d = tsne_3d.fit_transform(all_features)

    # Plot in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        tsne_features_3d[:, 0],
        tsne_features_3d[:, 1],
        tsne_features_3d[:, 2],
        c=[plot_colors[label] for label in all_labels],
    )

    legend1 = ax.legend(
        [
            plt.Line2D([0], [0], marker="o", color=color, linestyle="")
            for color in plot_colors
        ],
        plot_classes,
        loc="right",
    )
    ax.add_artist(legend1)
    plt.title(f"Class visualization with T-SNE (3D) exp {experiment}")

    # Save 3D visualization
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/latent_space_tsne_3D.png")

    # Optional: Animate 3D visualization
    angles = np.linspace(0, 360, 100)[:-1]
    rotanimate(
        ax, angles, "images/latent_space_tsne_3D.gif", delay=100, width=7, height=6
    )

    # TSNE for 2D Visualization
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_features_2d = tsne_2d.fit_transform(all_features)

    # Plot in 2D
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        tsne_features_2d[:, 0],
        tsne_features_2d[:, 1],
        c=[plot_colors[label] for label in all_labels],
    )
    legend1 = plt.legend(
        [
            plt.Line2D([0], [0], marker="o", color=color, linestyle="")
            for color in plot_colors
        ],
        plot_classes,
        loc="right",
    )
    plt.gca().add_artist(legend1)
    plt.title("Latent Space Visualization with T-SNE (2D)")
    plt.xlabel("T-SNE Component 1")
    plt.ylabel("T-SNE Component 2")

    # Save 2D visualization
    plt.savefig("images/latent_space_tsne_2D.png")
