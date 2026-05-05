#This file loads the image dataset using PyTorch. It also applies preprocessing and data augmentation,
# such as resizing, flipping, rotating, converting to tensors, and normalization.
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# All images are resized to 224x224.
# This keeps the input size consistent for the CNN.
IMAGE_SIZE = 224

# Batch size means the model trains on 32 images at a time.
BATCH_SIZE = 32

# These normalization values to help make training more stable
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# These transformations are applied only to the training images.
# The random changes help the model generalize better.
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Randomly flips some images left to right.
    # A cat or dog is still the same class if it faces the other direction.
    transforms.RandomHorizontalFlip(p=0.5),

    # Rotates images slightly.
    # This helps the model handle photos that are not perfectly straight.
    transforms.RandomRotation(10),

    # Converts the image into a PyTorch tensor.
    transforms.ToTensor(),

    # Normalizes pixel values so training is more stable.
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Validation and test transforms should not use random augmentation.
# We want evaluation to be consistent 
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def get_dataloaders(data_dir="data", batch_size=BATCH_SIZE):
    # ImageFolder automatically labels images based on folder names.
    # for exampl, data/train/cat and data/train/dog
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=test_transforms
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=test_transforms
    )

    # The training loader shuffles data so the model does not see images
    # in the same order every epoch.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Validation and test do not need shuffling because we are only evaluating.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes