from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from core.utils import build_transforms


# ----------------------------
# Helpers
# ----------------------------
def parse_classname(name: str) -> Tuple[str, str]:
    """
    'Tomato___Late_blight' -> ('Tomato', 'Late_blight')
    If no '___', treat all as species and disease='unknown'.
    """
    if "___" in name:
        sp, dz = name.split("___", 1)
    else:
        sp, dz = name, "unknown"
    return sp, dz


# ----------------------------
# ImageFolder variants
# ----------------------------
class ImageFolderWithPaths(datasets.ImageFolder):
    """Returns: (img, class_idx, path, index)"""

    def __getitem__(self, index):
        path, class_idx = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            class_idx = self.target_transform(class_idx)
        return img, class_idx, path, index


class MultiTaskImageFolder(datasets.ImageFolder):
    def __init__(self, root, healthy_keyword: str = "healthy", **kwargs):
        super().__init__(root=root, **kwargs)
        self.healthy_keyword = healthy_keyword.lower()

        # Build vocabularies from class folder names
        species_set, disease_set = set(), set()
        self.class_to_parts: Dict[str, Tuple[str, str]] = {}
        for cname in self.classes:
            sp, dz = parse_classname(cname)
            species_set.add(sp)
            disease_set.add(dz)
            self.class_to_parts[cname] = (sp, dz)

        self.species_list: List[str] = sorted(species_set)
        self.disease_list: List[str] = sorted(disease_set)  # includes 'healthy'
        self.species_to_id: Dict[str, int] = {s: i for i, s in enumerate(self.species_list)}
        self.disease_to_id: Dict[str, int] = {d: i for i, d in enumerate(self.disease_list)}

    def __getitem__(self, index):
        path, orig_class_idx = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        # Base class name (e.g., 'Tomato___Late_blight')
        cname = self.classes[orig_class_idx]
        sp, dz = self.class_to_parts[cname]

        species_idx = self.species_to_id[sp]
        disease_idx = self.disease_to_id[dz]
        # Convention: 1=Healthy, 0=Sick  (matches training code we discussed)
        health_idx = 1 if dz.lower() == self.healthy_keyword else 0

        target = {
            "species": torch.tensor(species_idx, dtype=torch.long),
            "health": torch.tensor(health_idx, dtype=torch.long),
            "disease": torch.tensor(disease_idx, dtype=torch.long),
            "orig_class": torch.tensor(orig_class_idx, dtype=torch.long),
        }
        return img, target, path, index

    # Convenience for stratification without iterating all samples
    def get_targets(self, which: str = "orig_class") -> np.ndarray:
        which = which.lower()
        vals = []
        for path, orig_class_idx in self.samples:
            cname = self.classes[orig_class_idx]
            sp, dz = self.class_to_parts[cname]
            if which == "orig_class":
                vals.append(orig_class_idx)
            elif which == "species":
                vals.append(self.species_to_id[sp])
            elif which == "disease":
                vals.append(self.disease_to_id[dz])
            elif which == "health":
                vals.append(1 if dz.lower() == self.healthy_keyword else 0)
            else:
                raise ValueError(f"Unknown target '{which}'")
        return np.array(vals, dtype=np.int64)


class FolderLabeledDataset(Dataset):
    def __init__(self, root: Path, img_size: int, mean: List[float], std: List[float],
                 multitask: bool, healthy_keyword: str = "healthy"):
        items = _iter_images_with_label_roots(root)
        if not items:
            raise FileNotFoundError(f"No images found under {root}")
        self.items = items
        self.multitask = multitask
        self.healthy_keyword = healthy_keyword
        self.tf = build_transforms(img_size, "val", mean, std)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, folder_label = self.items[idx]
        img = Image.open(p).convert("RGB")
        x = self.tf(img)

        if self.multitask:
            species, disease, health = _parse_species_disease(folder_label, self.healthy_keyword)
            truth = {"species": species, "disease": disease, "health": health}
        else:
            truth = {"combined": folder_label}

        return x, str(p), truth


def _iter_images_with_label_roots(root: Path) -> List[Tuple[Path, str]]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    items = []
    img_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for img_dir in img_dirs:
        all_imgs = sorted(img_dir.rglob("*"))
        for img in all_imgs:
            if img.suffix.lower() in extensions:
                items.append((img, img_dir.name))
    return items


def _parse_species_disease(label_dir_name: str, healthy_keyword: str = "healthy") -> Tuple[str, str, str]:
    parts = label_dir_name.split("___")
    if len(parts) == 2:
        species = parts[0]
        disease = parts[1]
        is_healthy = healthy_keyword.lower() in disease.lower()
        health_name = "Healthy" if is_healthy else "Sick"
        return species, disease, health_name
    else:
        raise ValueError(f"Invalid label_dir_name '{label_dir_name}'")


# ----------------------------
# Samplers (stratified)
# ----------------------------
def stratified_samplers(ds: datasets.ImageFolder,
                        val_split: float = 0.15,
                        seed: int = 42,
                        stratify_on: str = "species"):
    """
    Create train/val SubsetRandomSamplers with stratification.
    For MultiTaskImageFolder, you can stratify on {'orig_class','species','disease','health'}.
    For plain ImageFolder, it falls back to original targets.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    indices = np.arange(len(ds))

    if isinstance(ds, MultiTaskImageFolder):
        targets = ds.get_targets(stratify_on)
    else:
        targets = np.array(ds.targets)

    val_indices, train_indices = [], []
    for c in np.unique(targets):
        c_idx = indices[targets == c]
        rng.shuffle(c_idx)
        n_val = max(1, int(len(c_idx) * val_split))
        val_indices.extend(c_idx[:n_val])
        train_indices.extend(c_idx[n_val:])

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    return train_sampler, val_sampler


# ----------------------------
# Builders
# ----------------------------
def make_datasets(
        data_dir: Path, img_size: int, val_split: float = 0.15, seed: int = 42, multitask: bool = False,
        healthy_keyword: str = "healthy", stratify_on: str = "species"
):
    train_tf = build_transforms(img_size, "train")
    val_tf = build_transforms(img_size, "val")

    train_ds = MultiTaskImageFolder(root=str(data_dir), transform=train_tf, healthy_keyword=healthy_keyword)
    val_ds = MultiTaskImageFolder(root=str(data_dir), transform=val_tf, healthy_keyword=healthy_keyword)

    # For heads: len(train_ds.species_list), len(train_ds.disease_list), health=2
    # Stratify by species by default (better coverage than orig_class)
    train_sampler, val_sampler = stratified_samplers(train_ds, val_split, seed, stratify_on)

    # idx_to_class: original combined labels
    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return train_ds, val_ds, train_sampler, val_sampler, idx_to_class


def build_loaders(train_ds,
                  val_ds,
                  train_sampler,
                  val_sampler,
                  batch_size: int = 32,
                  num_workers: int = 4):
    """
    Default collate_fn can handle dict targets returned by MultiTaskImageFolder.
    """
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
