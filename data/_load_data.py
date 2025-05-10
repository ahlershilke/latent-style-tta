from data._datasets import get_dataset

def get_train_val_datasets():
    dataset = get_dataset(
        name='PACS',
        root_dir='/mnt/data/hahlers/datasets',
        test_domain=None,
        augment=None
    )
    
    # 80/20 split
    train_dataset = dataset.generate_train_dataset(val_ratio=0.2, stratify=True)
    val_dataset = dataset.generate_val_dataset(val_ratio=0.2, stratify=True)
    
    return train_dataset, val_dataset