import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineClassifier2d(nn.Module):
    """Baseline 2D CNN classifier for images."""
    def __init__(self, n_classes: int, channels: int = 1, input_size: int = 28) -> None:
        super().__init__()

        embedding_size = (input_size // 4) * (input_size // 4) * 64

        self.feature_extractor = nn.Sequential(
                nn.Conv2d(channels, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                )
        self.classifier = nn.Sequential(
                nn.Linear(embedding_size, 64),
                nn.BatchNorm1d(64),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(64, n_classes),
                )

    def forward(self, example: dict) -> torch.Tensor:
        x = example["data"]
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class ConvBlock(nn.Module):
    """Convolutional block for 2D CNN auto encoder."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, p: int=2):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                )
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=p, stride=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        out = self.downsample(out)
        return out

class DeconvBlock(nn.Module):
    """Transposed convolutional block for 2D CNN auto encoder."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, p: int=2):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=p, stride=p, padding=0)
        self.layers = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample(x)
        out = self.layers(out)
        return out

class BaselineAutoencoder2d(nn.Module):
    """Baseline 2D CNN based autoencoder for images."""
    def __init__(self, channels: int = 1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
                ConvBlock(channels, 4, kernel_size=3, padding=1, p=4), 
                ConvBlock(4, 4, kernel_size=3, padding=1, p=4), 
                ConvBlock(4, 8, kernel_size=3, padding=1, p=4),
                )

        self.decoder = nn.Sequential(
                DeconvBlock(8, 4, kernel_size=3, padding=1, p=4), 
                DeconvBlock(4, 4, kernel_size=3, padding=1, p=4), 
                DeconvBlock(4, 1, kernel_size=3, padding=1, p=4),
                )

    def forward(self, example: dict) -> torch.Tensor:
        x = example["data"]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Baseline1dCNNRegressor(nn.Module):
    """Node regressor based on 1D real-valued convolutions."""
    def __init__(self) -> None:
        super().__init__()
        kernel_size = 5
        padding = (kernel_size - 1) // 2
        self.feature_extractor = nn.Sequential(
                nn.Conv1d(2, 16, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.BatchNorm1d(16),
                nn.Conv1d(16, 32, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Conv1d(32, 64, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Conv1d(64, 64, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                )
        self.regression_head = nn.Sequential(
                nn.Linear(64, 1),
                )

    def forward(self, example: dict) -> torch.Tensor:
        x = example["data"] # (B, 1, L)
        x = torch.view_as_real(x) # (B, 1, L, 2)
        x = x.squeeze(1) # (B, L, 2)
        x = x.permute(0, 2, 1) # (B, 2, L)
        x = self.feature_extractor(x) # (B, 64, L)
        x = x.permute(0, 2, 1) # (B, L, 64)
        x = self.regression_head(x) # (B, L, 1)
        x = x.permute(0, 2, 1) # (B, 1, L)
        return x

def cycle_adj_with_self(n: int) -> torch.Tensor:
    """Adjacency matrix of cycle graph including self-loops and normalization."""
    A = torch.zeros(n, n)
    idx = torch.arange(n)
    A[idx, (idx - 1) % n] = 1
    A[idx, (idx + 1) % n] = 1
    A[idx, idx] = 1
    return A / 3.0

class GCNLayer(nn.Module):
    """Simple Graph Convolutional Layer (Kipf and Welling, 2017) for cycle graphs."""
    def __init__(self, in_features: int, out_features: int, n: int):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=False)
        self.register_buffer("A_hat", cycle_adj_with_self(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features, n)
        x = x.transpose(1, 2)           # (B, n, in_features)
        x = self.A_hat @ x              # (B, n, in_features)
        x = self.lin(x)                 # (B, n, out_features)
        return x.transpose(1, 2)        # (B, out_features, n)


class CycleGCNClassifier(nn.Module):
    """Baseline classifier based on graph convolution."""
    def __init__(self, n: int, in_dim: int, hidden_dim: int, n_classes: int, n_layers: int = 2):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(GCNLayer(in_dim, hidden_dim, n))
            in_dim = hidden_dim
        self.gcn = nn.ModuleList(layers)
        self.pool = lambda x: x.mean(dim=-1)  # global mean pooling
        self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.BatchNorm1d(64),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(64, n_classes)
                )

    def forward(self, example: torch.Tensor) -> torch.Tensor:
        x = example["data"] # (B, C, L) complex
        B, C, L = x.shape
        x = torch.view_as_real(x) # (B, C, L, 2) real
        x = x.permute(0, 3, 1, 2).reshape(B, 2 * C, L)
        for gcn in self.gcn:
            x = F.relu(gcn(x))
        x = self.pool(x)                # (B, hidden_dim)
        return self.classifier(x)       # (B, num_classes)

