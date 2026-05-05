#This file runs the full CNN training process. It loads the data, creates the CNN, 
# sets the loss function and optimizer, trains for 5 epochs, and saves the best model.
import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn_model import SimpleCNN
from utils.data import get_dataloaders
from utils.train_eval import train_one_epoch, evaluate

def main():
    # Use GPU if available, otherwise use CPU.
    # In my run, it used CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load train, validation, and test data.
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    print("Classes:", classes)

    # Create the CNN model.
    # Dropout is set to 0.5 to reduce overfitting.
    model = SimpleCNN(dropout=0.5).to(device)

    # Loss function for binary classification.
    # This works with one output logit(raw output values of final layer).
    criterion = nn.BCEWithLogitsLoss()

    # Adam optimizer updates the model weights.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Number of times the model sees the whole training set.
    num_epochs = 5

    # Track the best validation accuracy so we can save the best model.
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Train the model for one full epoch.
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate the model on validation data.
        # Validation data is not used to update weights.
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Print results so we can monitor learning.
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Save model only if validation accuracy improves.
        # This keeps the model that generalizes best.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_cnn.pth")
            print("Saved best model.")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()