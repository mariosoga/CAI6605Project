from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


class ImageFolderWithPaths(datasets.ImageFolder):
    """Overrides ImageFolder's default __getitem__ function to return:
     img, class_idx, path, and index"""

    def __getitem__(self, index):
        path, class_idx = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            class_idx = self.target_transform(class_idx)
        return img, class_idx, path, index


class MultiTaskImageFolder(datasets.ImageFolder):
    """Returns (img, disease_idx, health_idx, path, idx). health_idx: 0=healthy, 1=sick"""

    def __init__(self, root, healthy_keyword='healthy', **kwargs):
        super().__init__(root=root, **kwargs)
        self.healthy_keyword = healthy_keyword.lower()

    def __getitem__(self, index):
        path, disease_idx = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            disease_idx = self.target_transform(disease_idx)

        cls_name = self.classes[disease_idx].lower()
        health_idx = 0 if self.healthy_keyword in cls_name else 1
        return sample, disease_idx, health_idx, path, index


def build_transforms(img_size: int = 224):
    train_tf = v2.Compose([
        v2.Resize((img_size, img_size), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(10),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),

        # standard imagenet normalization
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_tf = v2.Compose([
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),

        # standard imagenet normalization
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return train_tf, val_tf


def stratified_samplers(ds: datasets.ImageFolder, val_split: float = 0.15, seed: int = 42):
    import numpy as np

    random = np.random.RandomState(seed)

    targets = np.array(ds.targets)
    indices = np.arange(len(ds))  # creates an array of all image indices

    val_indices, train_indices = [], []

    for c in np.unique(targets):
        c_indices = indices[targets == c]  # all indices of class c
        c_size = len(c_indices)

        random.shuffle(c_indices)  # random shuffle

        n_val = max(1, int(c_size * val_split))  # at least one sample per class goes to validation set

        val_indices.extend(c_indices[:n_val])
        train_indices.extend(c_indices[n_val:])

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    return train_sampler, val_sampler


def make_datasets(data_dir: Path, img_size: int, val_split: float = 0.15, seed: int = 42,
                  multitask: bool = False, healthy_keyword: str = 'healthy'):
    train_transform, val_transform = build_transforms(img_size)

    tmp_ds = datasets.ImageFolder(root=str(data_dir), transform=train_transform)

    class_to_idx = tmp_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_sampler, val_sampler = stratified_samplers(tmp_ds, val_split=val_split, seed=seed)

    if multitask:
        train_ds = MultiTaskImageFolder(root=str(data_dir), transform=train_transform, healthy_keyword=healthy_keyword)
        val_ds = MultiTaskImageFolder(root=str(data_dir), transform=val_transform, healthy_keyword=healthy_keyword)
    else:
        train_ds = ImageFolderWithPaths(root=str(data_dir), transform=train_transform)
        val_ds = ImageFolderWithPaths(root=str(data_dir), transform=val_transform)

    return train_ds, val_ds, train_sampler, val_sampler, idx_to_class


def build_loaders(train_ds, val_ds, train_sampler, val_sampler, batch_size: int = 32, num_workers: int = 4):
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,
                            pin_memory=True)
    return train_loader, val_loader
