import argparse
import json
import os

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np
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
        fname = "%s%03d.png" % (prefix, i)
        ax.figure.savefig(fname, dpi=ax.figure.dpi)
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

    folders = ["features/retrained", "features/salun_per_class"]

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
            # ----------------- TSNE data preparation ----------------------
            pre_tsne_feat[folder] = np.concatenate(
                [feat for _, feat in all_features[folder].items()]
            )
            all_labels_tsne[folder] = np.concatenate(
                [[i] * len(f) for i, f in all_features[folder].items()]
            )

        # ------------------ TSNE for 3D Visualization ---------------------
        tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
        tsne_features_3d = np.split(
            tsne_3d.fit_transform(
                np.concatenate([p for _, p in pre_tsne_feat.items()])
            ),
            len(pre_tsne_feat),
        )

        # ------------------ TSNE for 2D Visualization ---------------------
        tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_features_2d = np.split(
            tsne_2d.fit_transform(
                np.concatenate([p for _, p in pre_tsne_feat.items()])
            ),
            len(pre_tsne_feat),
        )

        for i, folder in enumerate(folders):
            class_separated = all_features[folder]
            os.makedirs(f"images/{folder}/tsne_3D/", exist_ok=True)
            os.makedirs(f"images/{folder}/tsne_2D/", exist_ok=True)
            os.makedirs(f"images/{folder}/tsne_animated/", exist_ok=True)
            # ----------------- TSNE plots creation -------------------
            # Plot in 3D
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(
                tsne_features_3d[i][:, 0],
                tsne_features_3d[i][:, 1],
                tsne_features_3d[i][:, 2],
                c=[plot_colors[label] for label in all_labels_tsne[folder]],
                s=1,
            )

            legend1 = ax.legend(
                [
                    plt.Line2D([0], [0], marker="o", color=color, linestyle="")
                    for color in plot_colors
                ],
                plot_classes,
                loc="center left",
                bbox_to_anchor=(1.05, 0.5),
            )
            ax.add_artist(legend1)
            plt.title(f"Latent Space {folder}_unlearned class{cls}")
            fig.tight_layout()
            # plt.show()
            # Save 3D visualization
            plt.savefig(f"images/{folder}/tsne_3D/class{cls}.png", dpi=fig.dpi)

            # Optional: Animate 3D visualization
            angles = np.linspace(0, 360, 100)[:-1]
            rotanimate(
                ax,
                angles,
                f"images/{folder}/tsne_animated/class{cls}.gif",
                delay=100,
                width=8,
                height=6,
            )
            plt.close(fig)

            # Plot in 2D
            plt.figure(figsize=(8, 6))
            fig = plt.figure()
            scatter = plt.scatter(
                tsne_features_2d[i][:, 0],
                tsne_features_2d[i][:, 1],
                c=[plot_colors[label] for label in all_labels_tsne[folder]],
                s=1,
            )
            plt.legend(
                [
                    plt.Line2D([0], [0], marker="o", color=color, linestyle="")
                    for color in plot_colors
                ],
                plot_classes,
                loc="center left",
                bbox_to_anchor=(1.05, 0.5),
            )
            plt.title(f"Latent Space {folder}_unlearned{cls}")
            fig.tight_layout()
            # plt.show()
            plt.savefig(f"images/{folder}/tsne_2D/class{cls}.png", dpi=fig.dpi)
            plt.close(fig)
            print(f"Saved images for class {cls} in experiment {folder}")
    print("Done")
