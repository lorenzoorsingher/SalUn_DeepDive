import torch
#import torchvision.models as models
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from datasets import UnlearnCifar10, UnlearnCifar100, UnlearnSVNH
from utils import load_checkpoint

if __name__ == '__main__':
    split = [0.7, 0.2, 0.1]
    model, config, transform, opt = load_checkpoint('checkpoints/resnet18_cifar10_pretrained_best.pt')
    dataset = UnlearnCifar10(split=split, transform=transform, unlearning_ratio=0)
    # samples = []
    # samples_to_retreive = 10
    # for i in dataset.classes:
    #     samples.append(dataset.get_samples(class_to_retreive = i, n_samples = samples_to_retreive))

    # load samples with the dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

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


    for data in dataloader:
        breakpoint()
        model(data['image'])
        latent_features = features[-1].squeeze().view(data['image'].size(0), -1)  # Flatten
        all_features.append(latent_features)
        all_labels.append(data['label'])

    all_features = torch.cat(all_features).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Apply PCA for 3D
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(all_features)

    # Plot in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_features[:, 0], pca_features[:, 1], pca_features[:, 2], c=all_labels, cmap='tab10')
    legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="Classes")
    ax.add_artist(legend1)
    plt.title('Latent Space Visualization (3D)')
    plt.show()


    # Apply PCA for 2D
    pca_2d = PCA(n_components=2)
    pca_features_2d = pca_2d.fit_transform(all_features)

    # Plot in 2D
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_features_2d[:, 0], pca_features_2d[:, 1], c=all_labels, cmap='tab10')
    legend1 = plt.legend(*scatter.legend_elements(), loc="best", title="Classes")
    plt.gca().add_artist(legend1)
    plt.title('Latent Space Visualization (2D)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
