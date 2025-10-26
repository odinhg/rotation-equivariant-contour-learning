import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import pad_contour, center_contour

class ConvLayer(nn.Module):
    """
    Complex-valued circular convolution layer. Input tensor of shape (B, in_channels, n). Output tensor of shape (B, out_channels, n).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.weight = nn.Parameter(
            torch.zeros(
                size=(out_channels, in_channels, kernel_size), dtype=torch.cfloat
            ),
            requires_grad=True,
        )
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = center_contour(x)

        if self.padding:
            x = pad_contour(x, size=self.padding)

        out = F.conv1d(x, self.weight, bias=None, stride=self.stride, padding=0, dilation=1)

        return out

class TransposedConvLayer(nn.Module):
    """
    Complex-valued circular transposed convolution layer. Input tensor of shape (B, in_channels, n). Output tensor of shape (B, out_channels, n).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.weight = nn.Parameter(
            torch.zeros(
                size=(in_channels, out_channels, kernel_size), dtype=torch.cfloat
            ),
            requires_grad=True,
        )
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = center_contour(x)

        if self.padding:
            x = pad_contour(x, size=self.padding)

        out = F.conv_transpose1d(x, self.weight, bias=None, stride=self.stride, padding=0, dilation=1)

        return out

class SpatialMaxPool(nn.Module):
    """
    Max pool in the spatial dimension. Supports returning indices for later unpooling.
    Input tensor of shape (B, C, n). Output tensor of shape (B, C, m) where m = n // p.
    """
    def __init__(self, p: int = 2) -> None:
        super().__init__()
        self.p = p

    def forward(
        self, x: torch.Tensor, return_indices: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor]:

        out, index = F.max_pool1d(x.abs(), self.p, stride=self.p, return_indices=True)
        out = torch.gather(x, dim=-1, index=index)

        if return_indices:
            return out, index

        return out


class SpatialMaxUnpool(nn.Module):
    """
    Max upool in the spatial dimension. Input tensor of shape (B, C, m). Output tensor of shape (B, C, n)
    """

    def __init__(self, p: int = 2) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        out_real = F.max_unpool1d(x.real, indices, self.p, self.p)
        out_imag = F.max_unpool1d(x.imag, indices, self.p, self.p)
        out = torch.stack([out_real, out_imag], dim=-1)
        out = torch.view_as_complex(out)
        return out

class SpatialAveragePool(nn.Module):
    """
    Average pool in the spatial dimension.
    Input tensor of shape (B, C, n). Output tensor of shape (B, C, m) where m = n // p.
    """
    def __init__(self, p: int = 2) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.unfold(-1, self.p, self.p).mean(-1)
        return out

class SpatialLearnablePool(nn.Module):
    """
    Learnable combination of average and max pooling.
    Input tensor of shape (B, C, n). Output tensor of shape (B, C, m) where m = n // p.
    """
    def __init__(self, p: int = 2) -> None:
        super().__init__()
        self.p = p
        self.avg_pool = SpatialAveragePool(p=p)
        self.max_pool = SpatialMaxPool(p=p)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.sigmoid(self.alpha)
        out = a * self.avg_pool(x) + (1 - a) * self.max_pool(x)
        return out

class SpatialMaxCosetPool(nn.Module):
    """
    Equivariant max pooling in the spatial dimension pooling over cosets (every p-th element).
    Input tensor of shape (B, C, n). Output tensor of shape (B, C, m) where m = n // p.
    """
    def __init__(self, p: int = 2) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, n = x.shape
        assert n % self.p == 0, "n must be divisible by p"
        m = n // self.p
        x = x.view(B, C, self.p, m).permute(0, 1, 3, 2)  # (B,C,m,p)
        # get argmax indices based on absolute value
        idx = x.abs().argmax(dim=-1, keepdim=True)
        return x.gather(-1, idx).squeeze(-1)  # (B,C,m)

class SpatialAverageCosetPool(nn.Module):
    """
    Equivariant average pooling in the spatial dimension pooling over cosets (every p-th element).
    Input tensor of shape (B, C, n). Output tensor of shape (B, C, m) where m = n // p.
    """
    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, n = x.shape
        assert n % self.p == 0, "n must be divisible by p"
        m = n // self.p

        # Reshape (B,C,n) -> (B,C,p,m) -> (B,C,m,p)
        x = x.view(B, C, self.p, m).permute(0, 1, 3, 2)
        return x.mean(dim=-1)  # (B,C,m)

class SpatialLearnableCosetPool(nn.Module):
    """
    Learnable combination of average and max coset pooling.
    Input tensor of shape (B, C, n). Output tensor of shape (B, C, m) where m = n // p.
    """
    def __init__(self, p: int = 2) -> None:
        super().__init__()
        self.p = p
        self.avg_pool = SpatialAverageCosetPool(p=p)
        self.max_pool = SpatialMaxCosetPool(p=p)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.sigmoid(self.alpha)
        out = a * self.avg_pool(x) + (1 - a) * self.max_pool(x)
        return out

class SpatialPool(nn.Module):
    """
    Spatial pooling layer that support maximum (magnitude) pooling, average pooling or a learnable combination of both.
    Input tensor of shape (B, C, n). Output tensor of shape (B, C, m) where m = n // p. 
    """
    def __init__(self, p: int = 2, method: str = "average", strided_pool: bool = True) -> None:
        super().__init__()
        self.p = p
        self.method = method

        if method == "average":
            self.pool = SpatialAveragePool(p=p) if strided_pool else SpatialAverageCosetPool(p=p)
        elif method == "max":
            self.pool = SpatialMaxPool(p=p) if strided_pool else SpatialMaxCosetPool(p=p)
        elif method == "learnable":
            self.pool = SpatialLearnablePool(p=p) if strided_pool else SpatialLearnableCosetPool(p=p)
        else:
            raise ValueError(f"Unknown pooling method: {method}. Supported methods: 'average', 'max', 'learnable'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Input tensor must be of shape (B, C, n), but got {x.shape}.")
        if x.shape[-1] % self.p != 0:
            raise ValueError(f"Input tensor length {x.shape[-1]} must be divisible by pooling size {self.p}.")
        return self.pool(x)


class ModReLU(nn.Module):
    """
    Rotation-equivariant activation function for complex-valued input based on ReLU.
    Input tensor of shape (B, C, n). Output tensor of shape (B, C, n).
    """
    def __init__(self) -> None:
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x.abs() + self.b) * x / x.abs()

class ModTanh(nn.Module):
    """
    Rotation-equivariant activation function for complex-valued input based on tanh.
    Input tensor of shape (B, C, n). Output tensor of shape (B, C, n).
    """
    def __init__(self) -> None:
        super().__init__()
        #self.b = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #return F.tanh(x.abs() + self.b) * x / x.abs()
        return F.tanh(x.abs()) * x / x.abs()

class SigLog(nn.Module):
    """
    Rotation-equivariant Siglog activation function for complex-valued input.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.abs() + 1)

class ActivationFunction(nn.Module):
    """
    Activation function layer that supports ModReLU, ModTanh and SigLog.
    Input tensor of shape (B, C, n). Output tensor of shape (B, C, n).
    """
    def __init__(self, method: str = "modrelu") -> None:
        super().__init__()
        if method == "modrelu":
            self.activation = ModReLU()
        elif method == "modtanh":
            self.activation = ModTanh()
        elif method == "siglog":
            self.activation = SigLog()
        else:
            raise ValueError(f"Unknown activation function: {method}. Supported methods: 'modrelu', 'modtanh', 'siglog'.")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)

class GlobalPool(nn.Module):
    """
    Global pooling layer outputs a single value per channel, aggregating absolute values across the spatial dimension. The output feature vector is shift and rotation invariant.
    Input tensor of shape (B, C, n). Output tensor of shape (B, C).
    """
    def __init__(self, in_channels: int, method: str = "learnable") -> None:
        super().__init__()
        self.in_channels = in_channels
        if method not in ["average", "max", "learnable"]:
            raise ValueError(f"Unknown pooling method: {method}. Supported methods: 'average', 'max', 'learnable'.")

        self.method = method

        if method == "learnable":
            self.alpha = nn.Parameter(torch.ones(size=(1, in_channels)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = center_contour(x)
        D = x.abs()
        if self.method == "average":
            out = D.mean(dim=-1)
        elif self.method == "max":
            out = D.max(dim=-1)[0]
        else:
            a = F.sigmoid(self.alpha)
            out = a * D.mean(dim=-1) + (1 - a) * D.max(dim=-1)[0]
        return out

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(num_features) * 0.1)
        self.beta = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        x = center_contour(x)
        magnitude = torch.abs(x)
        magnitude = torch.clamp(magnitude, min=1e-5)

        # Logarithmic normalization
        log_magnitude = torch.log(magnitude)
        log_batch_mean = log_magnitude.mean(dim=(0, -1), keepdim=True)
        log_batch_var = log_magnitude.var(dim=(0, -1), unbiased=False, keepdim=True)

        # Update running statistics
        if self.training:  
            self.running_mean = (
                self.momentum * log_batch_mean.squeeze() + (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * log_batch_var.squeeze() + (1 - self.momentum) * self.running_var
            )
        else:
            log_batch_mean = self.running_mean.view(1, -1, 1) 
            log_batch_var = self.running_var.view(1, -1, 1)


        log_magnitude_norm = (log_magnitude - log_batch_mean) / torch.sqrt(torch.clamp(log_batch_var, min=1e-6))

        gamma = self.gamma.view(1, -1, 1) 
        beta = self.beta.view(1, -1, 1)

        log_magnitude_scaled = gamma * log_magnitude_norm + beta
        magnitude_scaled = torch.exp(log_magnitude_scaled)

        out = magnitude_scaled * x / magnitude

        return out

