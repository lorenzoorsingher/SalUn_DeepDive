import os
import argparse
from utils import load_checkpoint
from datasets import get_dataloaders
import torch
from tqdm import tqdm
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Feature Extraction")

    parser.add_argument(
        "--directory",
        "-D",
        type=str,
        default="retrained",
    )
    parser.add_argument(
        "--samples",
        "-S",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--num-workers",
        "-W",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        "-B",
        type=int,
        default=32,
    )
    args = parser.parse_args()
    DIR = args.directory
    SAMPLES = args.samples
    NUM_WORKERS = args.num_workers
    BATCH_SIZE = args.batch_size

    for d in os.listdir(DIR):
        if not d.endswith(".pt"):
            continue
        CHECKPOINT = os.path.join(DIR, d)
        split = [0.7, 0.2, 0.1]
        model, config, transform, opt = load_checkpoint(CHECKPOINT)
        DSET = config["dataset"]
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
            batch_s=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )

        experiment = d.split("/")[-1].split(".")[0]
        # Load pretrained model
        model.to(DEVICE)
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
                img = data["image"].to(DEVICE)
                model(img)
                latent_features = (
                    features.pop().squeeze().view(img.size(0), -1).cpu()
                )  # Flatten
                all_features.append(latent_features)
                all_labels.append(data["label"])

                if idx == SAMPLES:
                    break

        all_features = torch.cat(all_features).numpy()
        all_labels = torch.cat(all_labels).numpy()

        os.makedirs(f"features/{args.directory}", exist_ok=True)

        with open(f"features/{args.directory}/{experiment}.json", "w") as f:
            json.dump(
                {
                    "all_features": all_features.tolist(),
                    "all_labels": all_labels.tolist(),
                },
                f,
            )
