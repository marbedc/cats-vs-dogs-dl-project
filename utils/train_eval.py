import torch

def train_one_epoch(model, loader, criterion, optimizer, device):
    # Set model to training mode.
    # This turns on training behavior 
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # Loop through the training data one batch at a time.
    for images, labels in loader:
        # Move images and labels to CPU 
        images = images.to(device)

        # BCEWithLogitsLoss expects labels to be floats
        # and shaped like [batch_size, 1].
        labels = labels.float().unsqueeze(1).to(device)

        # Clear old gradients before calculating new ones.
        optimizer.zero_grad()

        # Forward pass: get model predictions.
        outputs = model(images)

        # Calculate how wrong the predictions are.
        loss = criterion(outputs, labels)

        # Backpropagation: calculate gradients.
        loss.backward()

        # Update model weights using the optimizer.
        optimizer.step()

        # Keep track of total loss for this epoch.
        running_loss += loss.item() * images.size(0)

        # Convert raw logits into probabilities using sigmoid.
        # If probability >= 0.5, predict class 1.
        # Otherwise, predict class 0.
        preds = (torch.sigmoid(outputs) >= 0.5).float()

        # Count how many predictions were correct.
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Average loss and accuracy for the full epoch.
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    # Set model to evaluation mode.
    # This turns off dropout randomness.
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    # No gradients are needed during validation or testing.
    # This saves memory and makes evaluation faster.
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(outputs) >= 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Store labels and predictions for later metrics
            # like confusion matrix, precision, and F1.
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, all_labels, all_preds