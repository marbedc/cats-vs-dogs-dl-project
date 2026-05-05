#This file defines the CNN architecture. It contains the convolution layers that extract image features and the fully connected layers 
# #that classify the image as cat or dog.
import torch.nn as nn

#  goal: to classify an image as either a cat or a dog
class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()

        # This part of the network extracts image features
        # Each block uses:
        # 1. Convolution to learn visual patterns
        # 2. ReLU to add non-linearity
        # 3. MaxPool to reduce image size
        self.features = nn.Sequential(
            # First convolution block:
            # Input has 3 channels because images are RGB
            # Output has 32 feature maps, so the model learns 32 basic filters.
            # These early filters usually learn simple patterns like edges and colors.
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Max pooling reduces the image size from 224x224 to 112x112
            # This makes training faster and keeps the strongest features
            nn.MaxPool2d(kernel_size=2),

            # Second convolution block:
            # Takes the 32 feature maps from the previous layer and creates 64
            # This layer can learn more complex patterns like textures or shapes
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Reduces size from 112x112 to 56x56
            nn.MaxPool2d(kernel_size=2),

            # Third convolution block:
            # Increases feature maps from 64 to 128
            # Deeper layers can learn higher-level features like ears, faces, or fur patterns
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            # Reduces size from 56x56 to 28x28
            nn.MaxPool2d(kernel_size=2)
        )
        # This part classifies the extracted features as cat or dog
        self.classifier = nn.Sequential(
            # Flatten converts the 3D feature maps into one long vecto
            # After the three pooling layers, the size is 128 x 28 x 28
            nn.Flatten(),

            # Fully connected layer
            # It takes all extracted image features and combines them into 256 values
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),

            # Dropout randomly turns off some neurons during training
            # This helps prevent the model from memorizing the training data
            nn.Dropout(dropout),

            # Final output layer
            # Since this is binary classification, we only need one output value
            # This output is a raw logit, not a probability yet
            nn.Linear(256, 1)
        )
    # This defines how the image moves through the model
    def forward(self, x):
         # First, extract visual features from the image
        x = self.features(x)
        # Then, classify those features as cat or dog
        x = self.classifier(x)
        return x