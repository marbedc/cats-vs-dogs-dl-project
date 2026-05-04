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


def print_results(name, y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    print(f"\n===== {name} =====")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


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
