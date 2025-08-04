from data._datasets import get_dataset
from data._datasets import DomainSubset

"""
def get_train_val_datasets():
    dataset = get_dataset(
        name='PACS',
        root_dir='/mnt/data/hahlers/datasets',
        test_domain='photo',
        augment=None
    )
    
    # 80/20 split
    train_dataset, val_dataset = dataset.generate_train_val_datasets(val_ratio=0.2, stratify=True)

    # test
    test_domain_index = dataset.domains.index('photo') if 'photo' in dataset.domains else dataset.test_domain
    test_data = dataset.data[test_domain_index]
    
    return train_dataset, val_dataset, test_data
"""


def get_train_val_datasets():
    test_domain_name = 'photo'
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    #domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    test_domain_index = domains.index(test_domain_name)

    dataset = get_dataset(
        name='PACS',
        root_dir='/mnt/data/hahlers/datasets',
        test_domain=test_domain_index,
        augment=None
    )

    # 80/20 split
    train_dataset, val_dataset = dataset.generate_train_val_datasets(val_ratio=0.2, stratify=True)

    # Zugriff auf Testdaten direkt aus Originaldaten
    test_data = DomainSubset(
        dataset.data[test_domain_index],
        list(range(len(dataset.data[test_domain_index]))),
        domain_idx=len(dataset.data) - 1  # oder len(train_dataset.domains), wenn du das reindexed brauchst
    )

    return train_dataset, val_dataset, test_data


def get_lodo_splits():
    """Returns a list of datasets for Leave-One-Domain-Out (LoDO) cross-validation."""
    dataset = get_dataset(
        name='PACS',
        root_dir='/mnt/data/hahlers/datasets',
        test_domain=None,
        augment=None
    )
    return dataset.generate_lodo_splits()