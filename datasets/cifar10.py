import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_transforms(train=True):
    """Return appropriate transforms based on the training flag."""
    if train:
        return transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
        ])


def create_dataset(data_root, train, transform):
    """Create and return the CIFAR10 dataset."""
    return datasets.CIFAR10(root=data_root, train=train, download=True, transform=transform)


def create_dataloader(dataset, batch_size, is_train, **kwargs):
    """Create and return a DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=os.cpu_count() // 2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
        **kwargs
    )


def get_cifar10(batch_size, data_root='./', train=True, val=True, return_loader=False, **kwargs):
    loaders = []

    if train:
        train_dataset = create_dataset(data_root, True, create_transforms(train=True))
        train_loader = create_dataloader(train_dataset, batch_size, is_train=True, **kwargs)
        loaders.append(train_loader)

    if val:
        test_dataset = create_dataset(data_root, False, create_transforms(train=False))
        test_loader = create_dataloader(test_dataset, batch_size, is_train=False, **kwargs)
        loaders.append(test_loader)

    return loaders[0] if return_loader and len(loaders) == 1 else loaders if return_loader else None
