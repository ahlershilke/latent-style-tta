import os
import random
import torch
import numpy as np
from torchvision import transforms
from typing import Optional, Tuple

from typing import Any
from collections import defaultdict

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.datasets import ImageFolder
import PIL
from PIL import Image

"""
This code was taken and adapted from:
 https://github.com/VoErik/domain-generalization/blob/main/domgen/data/_datasets.py
"""

"""To add a new dataset, just create a class that inherits from `DomainDataset`."""

DOMAIN_NAMES = {
    'PACS': ["art_painting", "cartoon", "photo", "sketch"],
    'VLCS': ["Caltech101", "LabelMe", "SUN09", "VOC2007"]
}

class DomainSubset(Subset):
    """Wraps a dataset and adds domain index to the output."""
    def __init__(self, dataset, indices, domain_idx=None, original_domain_indices=None):
        super().__init__(dataset, indices)
        self.indices = list(indices) if not isinstance(indices, list) else indices
        self.original_domain_indices = original_domain_indices if original_domain_indices is not None else [domain_idx] * len(indices)
    
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label, self.original_domain_indices[idx]


class MultiDomainDataset:
    domains = None
    input_shape = None

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DomainDataset(MultiDomainDataset):
    def __init__(
            self,
            root: str,
            test_domain: Optional[int] = None,
            augment: Any = None,
            subset: float = None,
    ) -> None:
        """
        Base dataset class for a multi-domain dataset. Expects folder structure to be compatible with ImageFolder.
        :param root: Dataset directory.
        :param test_domain: Leave out domain.
        :param augment: Augment that needs to be applied. Defaults to ImageNet transformation.
        :param subset: Fraction of dataset that ought to be used. Keeps class and target distribution true to original
         data. Defaults to None = use entire dataset.
        :return: None
        """
        super().__init__()
        self.domains = sorted([directory.name for directory in os.scandir(root) if directory.is_dir()])
        self.test_domain = test_domain
        self.subset = subset
        self.data = []
        self.domain_to_idx = {domain: idx for idx, domain in enumerate(self.domains)}  # mapping domain -> index

        # base augment = ImageNet
        input_size = self.input_shape[-2], self.input_shape[-1]

        transform = transforms.Compose([
            transforms.Resize(input_size),
            #transforms.CenterCrop(input_size),
            transforms.ToTensor(),  # PIL -> Tensor + [0,1] normalisation
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        for i, domain in enumerate(self.domains):
            if augment and (i != self.test_domain):
                domain_transform = augment
            else:
                domain_transform = transform

            path = os.path.join(root, domain)
            domain_dataset = ImageFolder(path, transform=domain_transform)

            valid_samples = []
            for sample in domain_dataset.samples:
                try:
                    with Image.open(sample[0]) as img:
                        img.load()
                        img = img.convert("RGB")
                    valid_samples.append(sample)
                except (OSError, RuntimeError, PIL.UnidentifiedImageError):
                    print(f"Removing corrupted image: {sample[0]}")
            
            domain_dataset.samples = valid_samples
            domain_dataset.targets = [s[1] for s in valid_samples]


            if self.subset is not None:
                # ensures that target distribution remains true to original data
                num_samples_per_class = {}
                for target in set(domain_dataset.targets):
                    class_count = domain_dataset.targets.count(target)
                    num_samples_per_class[target] = int(class_count * self.subset)

                class_indices = defaultdict(list)
                for idx, target in enumerate(domain_dataset.targets):
                    class_indices[target].append(idx)

                subset_indices = []
                for target, indices in class_indices.items():
                    random.shuffle(indices)
                    subset_indices.extend(indices[:num_samples_per_class[target]])

                samples = [domain_dataset.samples[i] for i in subset_indices]
                targets = [domain_dataset.targets[i] for i in subset_indices]
                domain_dataset.samples = samples
                domain_dataset.targets = targets

            self.data.append(domain_dataset)

        self.num_classes = len(self.data[-1].classes)
        self.classes = list(self.data[-1].classes)
        self.idx_to_class = dict(zip(range(self.num_classes), self.classes))


    def get_domain_sizes(self) -> defaultdict:
        """Returns sizes of all domains."""
        size_dict = None
        domain_name_map = {i: name for i, name in enumerate(self.domains)}
        if self.data:
            size_dict = defaultdict()
            for i, domain_dataset in enumerate(self.data):
                size_dict[domain_name_map[i]] = len(domain_dataset.imgs)
        return size_dict


    def generate_loaders(
        self,
        batch_size: int = 32,
        test_size: float = 0.2,
        stratify: bool = True,
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Generates DataLoaders for training and testing domains with reindexed domain labels.
        :return: Tuple of (train_loader, val_loader, test_loader)
        """

        # custom collate function
        def collate_fn(batch):
            #batch = [item if len(item) == 3 else (*item, None) for item in batch]
            imgs = torch.stack([item[0] for item in batch])
            labels = torch.tensor([item[1] for item in batch])
            domains = torch.tensor([item[2] for item in batch])
            return imgs, labels, domains

        # Step 1: Create reindexing map
        train_domain_indices = [i for i in range(len(self.data)) if i != self.test_domain]
        domain_idx_mapping = {old: new for new, old in enumerate(train_domain_indices)}

        # Step 2: Split into train/val per domain using reindexed domain_idx
        train_subsets = []
        val_subsets = []

        for old_domain_idx in train_domain_indices:
            dom = self.data[old_domain_idx]
            targets = dom.targets

            train_idx, val_idx = train_test_split(
                np.arange(len(targets)),
                test_size=test_size,
                random_state=42,
                shuffle=True,
                stratify=targets if stratify else None
            )

            new_domain_idx = domain_idx_mapping[old_domain_idx]
            train_subsets.append(DomainSubset(dom, train_idx, new_domain_idx))
            val_subsets.append(DomainSubset(dom, val_idx, new_domain_idx))

        # Step 3: Create loaders
        train_loader = DataLoader(
            ConcatDataset(train_subsets),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            ConcatDataset(val_subsets),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        test_loader = None
        if self.test_domain is not None:
            test_domain_dataset = self.data[self.test_domain]
            """
            test_loader = DataLoader(
                DomainSubset(test_domain_dataset, list(range(len(test_domain_dataset))), domain_idx=len(train_domain_indices)),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            """
            test_subset = DomainSubset(
                test_domain_dataset,
                indices=list(range(len(test_domain_dataset))),
                domain_idx=len(train_domain_indices),
                original_domain_indices=[self.test_domain] * len(test_domain_dataset)
            )
            test_loader = DataLoader(
                test_subset,  # Verwende DomainSubset!
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0
            )
            print("Test loader exists in datasets:", test_loader is not None)
    
        return train_loader, val_loader, test_loader

    
    def __getitem__(self, index):
        """
        Returns (image, label, domain index) for the given index.
        """
        domain_idx = index[0]  # index of sub dataset (e.g. 0 for "art_painting")
        sample_idx = index[1]  # index in sub dataset
        img, label = self.data[domain_idx][sample_idx]
        
        return img, label, domain_idx
    

    def generate_train_val_datasets(self, val_ratio=0.2, stratify=True):
        """
        Generates train and validation datasets for all domains except the test domain.
        Applies domain index reindexing (0 ... N-1).
        """
        train_subsets = []
        val_subsets = []

        train_domain_indices = [i for i in range(len(self.data)) if i != self.test_domain]

        # index remapping
        domain_idx_mapping = {old: new for new, old in enumerate(train_domain_indices)}
        self.train_domain_idx_mapping = domain_idx_mapping  # optional: merken für später

        for old_domain_idx in train_domain_indices:
            dom = self.data[old_domain_idx]
            targets = dom.targets

            train_idx, val_idx = train_test_split(
                np.arange(len(targets)),
                test_size=val_ratio,
                stratify=targets if stratify else None,
                random_state=42
            )

            new_domain_idx = domain_idx_mapping[old_domain_idx]
            train_subsets.append(DomainSubset(dom, train_idx, new_domain_idx))
            val_subsets.append(DomainSubset(dom, val_idx, new_domain_idx))

        return ConcatDataset(train_subsets), ConcatDataset(val_subsets)


    def generate_lodo_splits(self):
        """Generates Leave-One-Domain-Out splits with consistent domain indices."""
        splits = []
    
        for test_domain_idx in range(len(self.domains)):
            train_domains = [i for i in range(len(self.domains)) if i != test_domain_idx]

            domain_idx_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(train_domains)}
            test_domain_new_idx = len(train_domains)

            train_subsets, val_subsets = [], []
            for orig_domain_idx in train_domains:
                domain_data = self.data[orig_domain_idx]
                targets = domain_data.targets
            
                # Train/Val-Split
                train_idx, val_idx = train_test_split(
                    np.arange(len(targets)),
                    test_size=0.2,
                    random_state=42,
                    stratify=targets
                )

                train_original_domains = [orig_domain_idx] * len(train_idx)
                val_original_domains = [orig_domain_idx] * len(val_idx)
            
                train_subsets.append(DomainSubset(
                    domain_data, 
                    train_idx, 
                    domain_idx=domain_idx_mapping[orig_domain_idx], 
                    original_domain_indices=train_original_domains
                ))
                val_subsets.append(DomainSubset(
                    domain_data, 
                    val_idx, 
                    domain_idx=domain_idx_mapping[orig_domain_idx], 
                    original_domain_indices=val_original_domains
                ))

            test_data = []
            test_data.append(DomainSubset(
                self.data[test_domain_idx],
                indices=list(range(len(self.data[test_domain_idx]))),
                domain_idx=len(train_domains),
                original_domain_indices=[test_domain_idx] * len(self.data[test_domain_idx])
            ))

            splits.append((
                ConcatDataset(train_subsets),  # Train
                ConcatDataset(val_subsets),    # Val
                ConcatDataset(test_data)       # Test
            ))
    
        return splits


def get_dataset(
        name: str,
        root_dir: str,
        test_domain: int,
        **kwargs,
) -> DomainDataset:
    """
    Gets a domain dataset from a given name.
    :param name: Dataset name as string. Must be one of: PACS, camelyon17
    :param root_dir: Path to datasets directory.
    :param test_domain: Leave out domain.
    :return:
    """
    if name == 'PACS':
        return PACS(root_dir, test_domain=test_domain, **kwargs)
    if name == 'VLCS':
        return VLCS(root_dir, test_domain=test_domain, **kwargs)
    else:
        raise ValueError(f"Dataset {name} not found. Please check the name or add it to the code.")


"""Insert new datasets below."""


class PACS(DomainDataset):
    domains = DOMAIN_NAMES['PACS']
    input_shape = (3, 227, 227)

    def __init__(self, root, test_domain, **kwargs):
        self.dir = os.path.join(root, "PACS/")
        self.aug = kwargs.get('augment', None)
        super().__init__(self.dir, test_domain, augment=self.aug)

        for domain_idx, domain_data in enumerate(self.data):
            domain_data.domain_idx = domain_idx

    def __getitem__(self, index):
        if isinstance(index, tuple):  # Nur für DomainSubset nötig
            domain_idx, sample_idx = index[0], index[1]
            img, label = self.data[domain_idx][sample_idx]
            return img, label, domain_idx
        else:  # Standardfall
            for domain_idx, domain_data in enumerate(self.data):
                if index < len(domain_data):
                    img, label = domain_data[index]
                    return img, label, domain_idx
                index -= len(domain_data)
            raise IndexError("Index out of range")
    
    def __len__(self):
        return sum(len(d) for d in self.data)


class VLCS(DomainDataset):
    domains = DOMAIN_NAMES['VLCS']
    input_shape = (3, 224, 224)

    def __init__(self, root, test_domain, **kwargs):
        self.dir = os.path.join(root, "VLCS/")
        self.aug = kwargs.get('augment', None)
        super().__init__(self.dir, test_domain, augment=None)

        for domain_idx, domain_data in enumerate(self.data):
            domain_data.domain_idx = domain_idx
    
    def __getitem__(self, index):
        if isinstance(index, tuple):  # Nur für DomainSubset nötig
            domain_idx, sample_idx = index[0], index[1]
            img, label = self.data[domain_idx][sample_idx]
            return img, label, domain_idx
        else:  # Standardfall
            for domain_idx, domain_data in enumerate(self.data):
                if index < len(domain_data):
                    img, label = domain_data[index]
                    return img, label, domain_idx
                index -= len(domain_data)
            raise IndexError("Index out of range")
    
    def __len__(self):
        return sum(len(d) for d in self.data)


    def check_corruption():
        root = "/mnt/data/hahlers/datasets/VLCS"
        corrupted = []

        for domain in os.listdir(root):
            dpath = os.path.join(root, domain)
            for fn in os.listdir(dpath):
                try:
                    img = Image.open(os.path.join(dpath, fn))
                    img.verify()
                except Exception:
                    corrupted.append(os.path.join(dpath, fn))

        print("Corrupted files:", corrupted)