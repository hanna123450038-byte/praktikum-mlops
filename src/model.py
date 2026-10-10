import torch
import torch.nn as nn
import torch.nn.functional as F


class FashionCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network designed for Fashion-MNIST classification.
    Equipped with configurable channels, fully connected units, and dropout toggling
    to support ablation studies and hyperparameter tuning.
    """

    def __init__(
        self,
        conv1_channels: int = 16,
        conv2_channels: int = 32,
        fc_units: int = 128,
        dropout_rate: float = 0.25,
        use_dropout: bool = True,
        num_classes: int = 10,
    ):
        super().__init__()
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.fc_units = fc_units
        self.dropout_rate = dropout_rate if use_dropout else 0.0
        self.use_dropout = use_dropout
        self.num_classes = num_classes

        # Feature extractor: 28x28 -> 14x14 -> 7x7
        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_channels)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_channels)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Classifier
        flatten_dim = conv2_channels * 7 * 7
        self.fc1 = nn.Linear(flatten_dim, fc_units)
        self.dropout = nn.Dropout(p=self.dropout_rate) if self.use_dropout and self.dropout_rate > 0 else nn.Identity()
        self.fc2 = nn.Linear(fc_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Flatten & dense
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    conv1_channels: int = 16,
    conv2_channels: int = 32,
    fc_units: int = 128,
    dropout_rate: float = 0.25,
    use_dropout: bool = True,
    num_classes: int = 10,
) -> FashionCNN:
    """Factory function for instantiating FashionCNN."""
    return FashionCNN(
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        fc_units=fc_units,
        dropout_rate=dropout_rate,
        use_dropout=use_dropout,
        num_classes=num_classes,
    )


if __name__ == "__main__":
    dummy_input = torch.randn(4, 1, 28, 28)
    model_with_drop = create_model(use_dropout=True, dropout_rate=0.3)
    model_without_drop = create_model(use_dropout=False)

    out1 = model_with_drop(dummy_input)
    out2 = model_without_drop(dummy_input)

    print(f"Output shape (with dropout): {out1.shape}")
    print(f"Output shape (without dropout): {out2.shape}")
    print(f"Total trainable params: {model_with_drop.count_parameters():,}")
