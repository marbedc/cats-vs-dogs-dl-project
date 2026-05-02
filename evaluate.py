import os
import json
import torch
import torch.nn as nn

from models.cnn_model import SimpleCNN
from utils.data import get_dataloaders
from utils.train_eval import evaluate
from utils.metrics import compute_metrics, save_confusion_matrix


RESULTS_DIR = "results"
CNN_CHECKPOINT = "best_cnn.pth"


def flatten_binary_outputs(values):
    flattened = []
    for x in values:
        if hasattr(x, "shape") and len(x.shape) > 0:
            flattened.append(int(x[0]))
        else:
            flattened.append(int(x))
    return flattened


def run_evaluation(model, model_name, checkpoint_path, test_loader, classes, device):
    if not os.path.exists(checkpoint_path):
        print(f"Skipping {model_name}: checkpoint '{checkpoint_path}' not found.")
        return None

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    test_loss, test_acc, all_labels, all_preds = evaluate(
        model, test_loader, criterion, device
    )

    all_labels = flatten_binary_outputs(all_labels)
    all_preds = flatten_binary_outputs(all_preds)

    metrics = compute_metrics(all_labels, all_preds)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save confusion matrix image
    cm_path = os.path.join(RESULTS_DIR, f"{model_name.lower()}_confusion_matrix.png")
    save_confusion_matrix(
        metrics["confusion_matrix"],
        classes,
        cm_path,
        title=f"{model_name} Confusion Matrix"
    )

    # Save metrics to JSON
    metrics_to_save = {
        "loss": test_loss,
        "accuracy": test_acc,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "confusion_matrix": metrics["confusion_matrix"].tolist()
    }

    json_path = os.path.join(RESULTS_DIR, f"{model_name.lower()}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_to_save, f, indent=4)

    print(f"\n===== {model_name} Results =====")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved metrics JSON to: {json_path}")

    return metrics_to_save


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    _, _, test_loader, classes = get_dataloaders()
    print("Classes:", classes)

    cnn_model = SimpleCNN(dropout=0.5)

    run_evaluation(
        model=cnn_model,
        model_name="CNN",
        checkpoint_path=CNN_CHECKPOINT,
        test_loader=test_loader,
        classes=classes,
        device=device
    )

    # Future extension for another model:
    # from models.pretrained_model import YourPretrainedModel
    # pretrained_model = YourPretrainedModel()
    # run_evaluation(
    #     model=pretrained_model,
    #     model_name="Pretrained",
    #     checkpoint_path="best_pretrained.pth",
    #     test_loader=test_loader,
    #     classes=classes,
    #     device=device
    # )


if __name__ == "__main__":
    main()
