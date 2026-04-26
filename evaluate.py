import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from models.pretrained_model import get_pretrained_model

# Paths
TEST_DIR = "/content/cats-vs-dogs-dl-project/data/test"

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_pretrained_model(num_classes=2)
model.load_state_dict(torch.load("best_pretrained.pth", map_location=device))
model = model.to(device)
model.eval()

# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Metrics
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
