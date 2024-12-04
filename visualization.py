import os

import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from datasets import UnlearnCifar10, UnlearnCifar100, UnlearnSVNH, get_dataloaders
from utils import load_checkpoint
from PIL import Image


def make_views(ax, angles, elevation=None, width=4, height=3,
               prefix='tmprot_', **kwargs):
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
        fname = '%s%03d.jpeg' % (prefix, i)
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
        :param loop:
    """
    # Open all images
    frames = [Image.open(file) for file in files]

    # Save the GIF
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],  # Include the other frames
        duration=delay,  # Duration of each frame in ms
        loop=loop  # Number of loops
    )


def rotanimate(ax, angles, output, **kwargs):
    """
    Produces an animation (.mp4,.ogv,.gif,.jpeg,.png) from a 3D plot on
    a 3D ax

    Args:
        ax (3D axis): the ax containing the plot of interest
        angles (list): the list of angles (in degree) under which to
                       show the plot.
        output : name of the output file. The extension determines the
                 kind of animation used.
        **kwargs:
            - width : in inches
            - heigth: in inches
            - framerate : frames per second
            - delay : delay between frames in milliseconds
            - repeat : True or False (.gif only)
    """

    output_ext = os.path.splitext(output)[1]

    files = make_views(ax, angles, **kwargs)

    D = {'.gif': make_gif}

    D[output_ext](files, output, **kwargs)

    for f in files:
        os.remove(f)


if __name__ == "__main__":
    split = [0.7, 0.2, 0.1]
    model, config, transform, opt = load_checkpoint(
        "checkpoints/resnet18_cifar10_pretrained_best.pt"
    )
    DSET = config["dataset"]

    (
        train_loader,
        val_loader,
        test_loader,
        forget_loader,
        retain_loader,
        _,
    ) = get_dataloaders(DSET, transform, unlr=0, itf=None, cf=None, batch_s=1, num_workers=2)

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
            latent_features = (features.pop().squeeze().view(data["image"].size(0), -1))  # Flatten
            all_features.append(latent_features)
            all_labels.append(data["label"])

            if idx == 200:
                break

    all_features = torch.cat(all_features).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Apply PCA for 3D
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(all_features)

    # Plot in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        pca_features[:, 0],
        pca_features[:, 1],
        pca_features[:, 2],
        c=all_labels,
        cmap="tab10",
    )
    legend1 = ax.legend(*scatter.legend_elements(), loc="right", title="Classes")
    ax.add_artist(legend1)
    plt.title("Class visualization")

    # plt.show()
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/latent_space_3D.png')
    angles = np.linspace(0, 360, 100)[:-1]  # Take 20 angles between 0 and 360
    # Remove the legend
    legend1.remove()
    rotanimate(ax, angles, 'images/latent_space_3D.gif', delay=100, width=7, height=6)

    # Apply PCA for 2D
    pca_2d = PCA(n_components=2)
    pca_features_2d = pca_2d.fit_transform(all_features)

    # Plot in 2D
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        pca_features_2d[:, 0], pca_features_2d[:, 1], c=all_labels, cmap="tab10"
    )
    legend1 = plt.legend(*scatter.legend_elements(), loc="best", title="Classes")
    plt.gca().add_artist(legend1)
    plt.title("Latent Space Visualization (2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig('images/latent_space_2d.png')
