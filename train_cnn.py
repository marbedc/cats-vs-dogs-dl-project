import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn_model import SimpleCNN
from utils.data import get_dataloaders
from utils.train_eval import train_one_epoch, evaluate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print("Classes:", classes)

    model = SimpleCNN(dropout=0.5).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_cnn.pth")
            print("Saved best model.")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()