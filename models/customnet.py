from torch import nn
import torch

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        # 3 x 3 convs with padding=1 keep H x W, so only the pools change the size.
        # bias=False: the BatchNorm right after each conv adds its own shift
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, bias=False)  # 1 x 1: mixes channels only (128 -> 64), so no padding
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(64, 256, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.pool5 = nn.AdaptiveAvgPool2d(1)  # global average: one value per channel, whatever the input size

        self.fc1 = nn.Linear(256, 200)      # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass

        # starting from B x 3 x 64 x 64
        x = self.conv1(x)                           # B x 64 x 64 x 64
        x = self.batch_norm1(x).relu()
        x = self.pool1(x)                           # B x 64 x 32 x 32

        x = self.conv2(x)                           # B x 128 x 32 x 32
        x = self.batch_norm2(x).relu()
        x = self.pool2(x)                           # B x 128 x 16 x 16

        x = self.conv3(x)                           # B x 64 x 16 x 16
        x = self.batch_norm3(x).relu()
        x = self.pool3(x)                           # B x 64 x 8 x 8

        x = self.conv4(x)                           # B x 64 x 8 x 8
        x = self.batch_norm4(x).relu()
        x = self.pool4(x)                           # B x 64 x 4 x 4

        x = self.conv5(x)                           # B x 256 x 4 x 4
        x = self.batch_norm5(x).relu()
        x = self.pool5(x)                           # B x 256 x 1 x 1

        x = torch.flatten(x, 1, -1)                 # B x 256

        x = self.fc1(x)                             # B x 200 - no relu() because the output is the logits
                                                              # that will be used by the CrossEntropyLoss

        return x