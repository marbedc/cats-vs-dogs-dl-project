import torch
import torch.nn as nn

from models.cnn_model import SimpleCNN
from models.pretrained_model import get_pretrained_model
from models.transformer_model import get_vit_model
from utils.data import get_dataloaders


def evaluate_binary_model(model, checkpoint, loader, device):
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs).squeeze() >= 0.5).long()

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred


def evaluate_multiclass_model(model, checkpoint, loader, device):
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred


import matplotlib.pyplot as plt
import seaborn as sns
import os

def print_results(name, y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n===== {name} =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["cat", "dog"],
        yticklabels=["cat", "dog"],
        annot_kws={"size": 16, "weight": "bold"}
    )

    plt.title(f"{name} Confusion Matrix\nAccuracy: {acc:.2%} | F1-score: {f1:.2f}", fontsize=14, weight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()

    filename = name.lower().replace(" ", "_").replace("-", "") + "_confusion_matrix.png"
    path = os.path.join("results", filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion matrix to {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    _, _, test_loader, classes = get_dataloaders()
    print("Classes:", classes)

    # 1. CNN
    cnn = SimpleCNN(dropout=0.5)
    y_true, y_pred = evaluate_binary_model(cnn, "best_cnn.pth", test_loader, device)
    print_results("CNN", y_true, y_pred)

    # 2. Pretrained ResNet
    pretrained = get_pretrained_model(num_classes=2)
    y_true, y_pred = evaluate_multiclass_model(pretrained, "best_pretrained.pth", test_loader, device)
    print_results("Pretrained ResNet", y_true, y_pred)

    # 3. Transformer (ViT)
    transformer = get_vit_model()
    y_true, y_pred = evaluate_binary_model(transformer, "best_transformer.pth", test_loader, device)
    print_results("Vision Transformer", y_true, y_pred)


if __name__ == "__main__":
    main()
