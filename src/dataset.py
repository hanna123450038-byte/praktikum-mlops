import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def download_fashion_mnist(data_dir: str = "./data") -> None:
    """Download Fashion-MNIST train and test sets if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    datasets.FashionMNIST(root=data_dir, train=True, download=True)
    datasets.FashionMNIST(root=data_dir, train=False, download=True)


def compute_file_md5(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute MD5 checksum for an individual file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dataset_hash(data_dir: str = "./data") -> str:
    """
    Compute deterministic MD5 hash for the dataset directory.
    Combines the MD5 hashes of all data files sorted by relative path.
    """
    path = Path(data_dir)
    if not path.exists():
        download_fashion_mnist(data_dir)

    all_files = sorted([f for f in path.rglob("*") if f.is_file()])
    if not all_files:
        download_fashion_mnist(data_dir)
        all_files = sorted([f for f in path.rglob("*") if f.is_file()])

    master_hasher = hashlib.md5()
    for file_path in all_files:
        # Include relative path and file MD5 to ensure tree structure and content tracking
        rel_path = file_path.relative_to(path).as_posix()
        file_hash = compute_file_md5(file_path)
        master_hasher.update(rel_path.encode("utf-8"))
        master_hasher.update(file_hash.encode("utf-8"))

    return master_hasher.hexdigest()


def get_transforms(use_augmentation: bool = False) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get train and test torchvision transforms with optional augmentation."""
    norm = transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST mean and std

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomCrop(28, padding=2, padding_mode="edge"),
            transforms.ToTensor(),
            norm,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            norm,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        norm,
    ])

    return train_transform, test_transform


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    use_augmentation: bool = False,
    subset_size: Optional[int] = None,
    val_split: float = 0.1,
    random_seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare and return train, validation, and test DataLoaders.
    Optionally subsets the training set for faster tuning.
    """
    train_transform, test_transform = get_transforms(use_augmentation=use_augmentation)

    full_train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    if subset_size is not None and subset_size < len(full_train_dataset):
        generator = torch.Generator().manual_seed(random_seed)
        indices = torch.randperm(len(full_train_dataset), generator=generator)[:subset_size]
        base_dataset = Subset(full_train_dataset, indices)
    else:
        base_dataset = full_train_dataset

    # Split into train and validation
    val_size = int(len(base_dataset) * val_split)
    train_size = len(base_dataset) - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_subset, val_subset = random_split(
        base_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def export_dataset_info(data_dir: str = "./data", output_path: str = "./data/metadata.json") -> dict:
    """Compute and save metadata for DVC tracking and data reproducibility."""
    download_fashion_mnist(data_dir)
    data_hash = compute_dataset_hash(data_dir)

    train_ds = datasets.FashionMNIST(root=data_dir, train=True)
    test_ds = datasets.FashionMNIST(root=data_dir, train=False)

    metadata = {
        "dataset_name": "Fashion-MNIST",
        "hash_md5": data_hash,
        "num_train_samples": len(train_ds),
        "num_test_samples": len(test_ds),
        "classes": CLASSES,
        "input_shape": [1, 28, 28],
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    print("Downloading Fashion-MNIST and generating metadata...")
    info = export_dataset_info()
    print(f"Dataset MD5 Hash: {info['hash_md5']}")
    print(f"Metadata saved successfully: {info}")
