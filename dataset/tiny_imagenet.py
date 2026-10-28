import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch

# Train: random crop (padding 4) and horizontal flip as augmentation, then ImageNet mean/std
transform_train = T.Compose([
    T.Resize((64, 64)),  # Resize to fit the input dimensions of the network
    T.RandomCrop(64, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Eval: same resize and normalization, nothing random, so every evaluation sees the same images
transform_eval = T.Compose([
    T.Resize((64, 64)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_dataloaders(root='data/tiny-imagenet-200', batch_size=32):
    # root is relative to the repo root: launch the scripts from there (as lab3.ipynb does)
    tiny_imagenet_dataset_train = ImageFolder(root=os.path.join(root, 'train'), transform=transform_train)
    tiny_imagenet_dataset_val = ImageFolder(root=os.path.join(root, 'val'), transform=transform_eval)

    # A val/ that was never reorganized is read as a single class, 'images': fail here, not with a meaningless val acc
    assert tiny_imagenet_dataset_val.class_to_idx == tiny_imagenet_dataset_train.class_to_idx, \
        'val/ is not split into class folders: run utils/download_dataset.py'

    print(f"Length of train dataset: {len(tiny_imagenet_dataset_train)}")
    print(f"Length of val dataset: {len(tiny_imagenet_dataset_val)}")

    # 2 worker processes prepare the batches in parallel and stay alive across epochs
    train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
