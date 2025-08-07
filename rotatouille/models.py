import torch
import torch.nn as nn
from .layers import ConvLayer, TransposedConvLayer, SpatialPool, ModReLU, ModTanh, GlobalPool, BatchNorm

class ConvBlock(nn.Module):
    """
    Convolutional block with a convolutional layer, ModReLU activation, and spatial pooling.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, p: int, method: str) -> None: 
        super().__init__()
        padding = (kernel_size - 1) // 2 # Keep output size the same as input size
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, padding)
        self.activation = ModReLU()
        self.bn = BatchNorm(out_channels)
        self.pool = SpatialPool(p=p, method=method) if p > 1 else nn.Identity()
        self.global_pool = GlobalPool(out_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.conv(x)
        out = self.activation(out)
        out = self.bn(out)
        out = self.pool(out)
        out_invariant = self.global_pool(out)
        return out, out_invariant

class FeatureExtractor(nn.Module):
    """
    Feature extractor with multiple convolutional blocks. Returns invariant feature vector.
    The architecture is defined by a list of dictionaries, where each dictionary contains the parameters for a convolutional block. For example:
    [
        {"in_channels": 1, "out_channels": 8, "kernel_size": 3, "p": 2, "method": "learnable"},
        {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "p": 2, "method": "learnable"},
    ]
    """
    def __init__(self, layers: list[dict]) -> None: 
        super().__init__()
        self.layers = nn.ModuleList()
        for layer in layers:
            self.layers.append(ConvBlock(**layer))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for layer in self.layers:
            x, invariant = layer(x)
            features.append(invariant)
        out = torch.cat(features, dim=-1)
        return out


class ContourClassifier(nn.Module):
    """
    Rotation invariant model for contour classification using equivariant layers. Supports additional features for the classifier head such as radial histogram values. The feature extractor is defined by a list of dictionaries, where each dictionary contains the parameters for a convolutional block. For example:

    [
        {"in_channels": 1, "out_channels": 8, "kernel_size": 3, "p": 2, "method": "learnable"},
        {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "p": 2, "method": "learnable"},
        {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "p": 2, "method": "learnable"},
    ]
    In this example, the feature vector will have a dimension of 8 + 16 + 32 = 56. If `extra_features_dim` is set to 10, the final embedding dimension will be 66.
    """
    def __init__(self, n_classes: int, feature_extractor_layers: list[dict], extra_features_dim: int = 0, fcnn_hidden_dim: int = 64, fcnn_dropout: float = 0.5) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(feature_extractor_layers)

        embedding_dim = sum(layer["out_channels"] for layer in feature_extractor_layers) + extra_features_dim
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, fcnn_hidden_dim),
            nn.BatchNorm1d(fcnn_hidden_dim),
            nn.Dropout(fcnn_dropout),
            nn.ReLU(),
            nn.Linear(fcnn_hidden_dim, n_classes),
        )


    def forward(self, example: dict) -> torch.Tensor:
        x = example["data"]
        invariant_features = self.feature_extractor(x)

        if "extra_features" in example: 
            extra_features = example["extra_features"]
            invariant_features = torch.cat([invariant_features, extra_features], dim=-1)

        out = self.classifier(invariant_features)
        return out

class AEConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, p: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layers = nn.Sequential(
                ConvLayer(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                BatchNorm(out_channels),
                ModTanh(),
                ConvLayer(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                BatchNorm(out_channels),
                ModTanh(),
                )

        if p > 1:
            self.pooling = ConvLayer(out_channels, out_channels, kernel_size=p, stride=p, padding=0)
        else:
            self.pooling = None

        self.skip_connection = nn.Identity() if in_channels == out_channels else ConvLayer(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residual = self.skip_connection(x)
        out_equivariant = self.layers(x) + residual
        if self.pooling is not None:
            out_equivariant = self.pooling(out_equivariant)
        return out_equivariant

class AEDeconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, p: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
                ConvLayer(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                ModTanh(),
                ConvLayer(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                ModTanh(),
                )

        if p > 1:
            self.unpooling = TransposedConvLayer(in_channels, in_channels, kernel_size=p, stride=p, padding=0)
        else:
            self.unpooling = None

        self.skip_connection = nn.Identity() if in_channels == out_channels else ConvLayer(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x_equivariant):
        if self.unpooling is not None:
            x_equivariant = self.unpooling(x_equivariant)
        residual = self.skip_connection(x_equivariant)
        x_equivariant = self.conv(x_equivariant) + residual
        return x_equivariant

class ShapeAutoEncoder(nn.Module):
    """
    Rotation equivariant autoencoder for contour data. The layers in the encoder are specified as a list of dictionaries, e.g.,
    [
        {"in_channels": 1, "out_channels": 8, "kernel_size": 3, "p": 2, "method": "learnable"},
        {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "p": 2, "method": "learnable"},
        {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "p": 2, "method": "learnable"},
    ]
    The decoder is defined by reversing this list leading to a symmetric architecture.
    """
    def __init__(self, layers: list[dict]) -> None: 
        super().__init__()

        self.encoder = nn.ModuleList()
        for layer in layers:
            self.encoder.append(AEConvBlock(**layer))

        self.decoder = nn.ModuleList()
        for layer in reversed(layers):
            layer["in_channels"], layer["out_channels"] = layer["out_channels"], layer["in_channels"]
            self.decoder.append(AEDeconvBlock(**layer))

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x

    def forward(self, example):
        x = example["data"]
        y = self.encode(x)
        z = self.decode(y)
        return z

class NodeRegressionModel(nn.Module):
    """
    Node-level regression. Input shape (B, 1, n) complex tensor, where B is the batch size and n is the number of nodes in the contour. Output shape (B, 1, n) real tensor (e.g., predicted curvature values).
    """
    def __init__(self) -> None:
        super().__init__()
        kernel_size = 5
        padding = (kernel_size - 1) // 2
        self.complex_layers = nn.Sequential(
            ConvLayer(1, 8, kernel_size=kernel_size, padding=padding),
            ModReLU(),
            BatchNorm(8),
            ConvLayer(8, 16, kernel_size=kernel_size, padding=padding),
            ModReLU(),
            BatchNorm(16),
            ConvLayer(16, 32, kernel_size=kernel_size, padding=padding),
            ModReLU(),
            BatchNorm(32),
            ConvLayer(32, 64, kernel_size=kernel_size, padding=padding),
            ModReLU(),
            BatchNorm(64),
            )
        # (B, 32, n) complex -- abs() --> (B, 32, n) real
        # Then we can use a real-valued regression head to predict the curvature values at each point
        self.regression_head = nn.Sequential(
            nn.Linear(64, 1),
            )

    def forward(self, example: dict) -> torch.Tensor:
        x = example["data"]
        B, n = x.shape[:2]  # B is the batch size, n is the number of nodes
        x = self.complex_layers(x)
        x = x - x.mean(dim=-1, keepdim=True)
        x = x.abs()
        x = x.permute(0, 2, 1)  # (B, n, 32)
        x = self.regression_head(x)
        x = x.permute(0, 2, 1)  # (B, 1, n)
        return x

        
                


if __name__ == "__main__":
    def rotate_batch(x: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        # Rotate a batch of contours by a random angles
        w = torch.polar(torch.ones(x.shape[0]), thetas).reshape(-1, 1, 1)
        return w * x 

    # Test model and invariance to rotation
    model = ContourClassifier(
        n_classes=4,
        feature_extractor_layers=[
            {"in_channels": 1, "out_channels": 8, "kernel_size": 3, "p": 2, "method": "learnable"},
            {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "p": 2, "method": "learnable"},
            {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "p": 2, "method": "learnable"},
        ],
        extra_features_dim=10
    )
    model.eval()

    x = torch.view_as_complex(torch.randn(8, 1, 128, 2))
    extra_features = torch.randn(8, 10)
    y = torch.randint(0, 4, (8,))
    example = {"data": x, "label": y, "extra_features": extra_features}
    with torch.no_grad():
        out = model(example)
    print(f"Input shape: {x.shape}, output shape: {out.shape}")

    # With random rotation of the input
    thetas = torch.rand(x.shape[0]) * 2 * torch.pi
    x_rot = rotate_batch(x, thetas)
    example_rot = {"data": x_rot, "label": y, "extra_features": extra_features}
    with torch.no_grad():
        out_rot = model(example_rot)
    print(f"Input shape: {x_rot.shape}, output shape: {out_rot.shape}")
    max_abs_diff = torch.max(torch.abs(out - out_rot))
    print(f"Invariance: max absolute difference between rotated and non-rotated input: {max_abs_diff}")
    assert torch.allclose(out, out_rot, atol=1e-4), "Invariance failed"

    # Test equivariance of ConvBlock to rotation
    conv_block = ConvBlock(1, 8, 3, 2, "learnable")
    conv_block.eval()
    with torch.no_grad():
        out = conv_block(x)
        out_rot = conv_block(x_rot)
    rot_out = rotate_batch(out[0], thetas)
    max_abs_diff = torch.max(torch.abs(out_rot[0] - rot_out))
    print(f"Equivariance: max absolute difference between rotate-then-conv and conv-then-rotate: {max_abs_diff}")
    assert torch.allclose(out_rot[0], rot_out, atol=1e-4), "Equivariance failed"

