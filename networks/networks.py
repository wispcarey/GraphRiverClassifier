import numpy as np

from torch import nn
import torch.nn.functional as F

class BasicAutoEncoder(nn.Module):
    def __init__(self, input_dim=726, bottleneck_dim=32, encode_layers=2):
        super(BasicAutoEncoder, self).__init__()

        # Encoder
        encoder_layers = []
        if encode_layers == 2:
            encoder_layers = [
                nn.Linear(input_dim, 128),
                nn.ReLU(True),
                nn.Linear(128, bottleneck_dim),
            ]
        elif encode_layers == 3:
            encoder_layers = [
                nn.Linear(input_dim, 128),
                nn.ReLU(True),
                nn.Linear(128, 32),
                nn.ReLU(True),
                nn.Linear(32, bottleneck_dim),
            ]
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        if encode_layers == 2:
            decoder_layers = [
                nn.Linear(bottleneck_dim, 128),
                nn.ReLU(True),
                nn.Linear(128, input_dim)
            ]
        elif encode_layers == 3:
            decoder_layers = [
                nn.Linear(bottleneck_dim, 32),
                nn.ReLU(True),
                nn.Linear(32, 128),
                nn.ReLU(True),
                nn.Linear(128, input_dim)
            ]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        return F.sigmoid(encoded)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decoder(encoded)
        return decoded, F.sigmoid(encoded)

class MLPEmbedding(nn.Module):
    def __init__(self, input_dim=726, bottleneck_dim=32, encode_layers=2):
        super(BasicAutoEncoder, self).__init__()

        # Encoder
        encoder_layers = []
        if encode_layers == 2:
            encoder_layers = [
                nn.Linear(input_dim, 128),
                nn.ReLU(True),
                nn.Linear(128, bottleneck_dim),
            ]
        elif encode_layers == 3:
            encoder_layers = [
                nn.Linear(input_dim, 128),
                nn.ReLU(True),
                nn.Linear(128, 32),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(32, bottleneck_dim),
            ]
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = F.normalize(self.encoder(x), dim=1)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, n, bottleneck_dim=32):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5,5), padding=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3,3), padding=1)

        self.size_after_conv = n // 2
        fc_input_dim = self.size_after_conv * self.size_after_conv * 24

        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.fc2 = nn.Linear(128, bottleneck_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.normalize(self.fc2(x), dim=1)
        return x