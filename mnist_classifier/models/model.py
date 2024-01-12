import torch.nn as nn
import torch.nn.functional as F


class MyNeuralNet(nn.Module):
    """Basic neural network class."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.flatten = nn.Flatten()
        self.output = nn.Linear(64 * 12 * 12, 10)
        self.leaky_relu = nn.LeakyReLU()
        self.max_pool2d = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: input tensor of shape [B, 1, 28, 28]

        Returns:
            Output tensor

        """
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.flatten(x)
        x = F.log_softmax(self.output(x), dim=1)
        return x
