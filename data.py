import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision import datasets, transforms
from torchtoolbox.transform import Cutout

def get_transform(input_size=224, is_val = False):
    if is_val:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5])
        ])

    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ColorJitter(0.45, 0.45, 0.45),

        transforms.RandomCrop(input_size, padding=16),
        Cutout(0.5),

        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5])
    ])

def get_loaders(dataroot, train_batch_size, test_batch_size, input_size, workers, data_val_path):
    train_data = datasets.ImageFolder(
        # root=os.path.join(dataroot, 'train'),
        root=os.path.join(dataroot, 'train_upper'),
        transform=get_transform(input_size=input_size))
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=train_batch_size, 
        shuffle=True,
        num_workers=workers, pin_memory=True)

    test_data = datasets.ImageFolder(
        root=os.path.join(dataroot, 'val_upper'), 
        # root=os.path.join(dataroot, 'train'),

        transform=get_transform(input_size=input_size))
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=test_batch_size * 2, 
        shuffle=True, 
        num_workers=workers,
        pin_memory=True)
    
    val_data = datasets.ImageFolder(
        root=data_val_path, 
        transform=get_transform(input_size=input_size, is_val = True))
    val_loader = torch.utils.data.DataLoader(
        val_data, 
        batch_size=test_batch_size * 1, 
        shuffle=True, 
        num_workers=workers,
        pin_memory=True)
    
    return train_loader, test_loader, val_loader
