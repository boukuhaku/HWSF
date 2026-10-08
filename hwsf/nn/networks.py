"""Neural networks of HWSF (Figs. 6 and 7, Supplementary Table S1).

All networks act on a leading batch dimension. Weather inputs have shape (N, 121, C) with days on the second axis.
"""
import torch
from torch import nn
from torch.nn import functional as F

from ..constants import CORRECTION_BOUND, MONTHS


def gelu(x):
    return F.gelu(x, approximate="none")


class RegionalNetwork(nn.Module):
    """Regional network: 26 standardized inputs (latitude, longitude, 24 April-July monthly climate means)
    -> 16-element regional representation and a scalar regional baseline in standardized yield units."""

    def __init__(self, in_features=26, width=16):
        super().__init__()
        self.hidden1 = nn.Linear(in_features, width)
        self.hidden2 = nn.Linear(width, width)
        self.output = nn.Linear(width, 1)

    def forward(self, s):
        h = torch.tanh(self.hidden1(s))
        h = torch.tanh(self.hidden2(h))
        return h, self.output(h)[..., 0]


class TemporalResidualBlock(nn.Module):
    """Kernel-7 temporal convolution (depthwise or full-channel), GELU, 1x1 convolution, GELU, plus the input."""

    def __init__(self, channels=12, kernel_size=7, dilation=1, depthwise=False):
        super().__init__()
        self.temporal = nn.Conv1d(channels, channels, kernel_size, padding=dilation * (kernel_size // 2),
                                  dilation=dilation, groups=channels if depthwise else 1)
        self.pointwise = nn.Conv1d(channels, channels, 1)

    def forward(self, z):
        return z + gelu(self.pointwise(gelu(self.temporal(z))))


class WeatherNetwork(nn.Module):
    """Annual-weather network (depthwise, no FiLM) or local-weather network (full-channel, with FiLM).

    A 1x1 convolution projects the daily inputs to 12 channels, three residual blocks with dilation rates 1, 2, and 4
    follow, and the monthly means of the last block (48 values) are joined with the 16 regional features. A dense head
    maps them to a 16-element weather representation and a scalar weather prediction (standardized yield units).
    With FiLM, a block-specific dense layer maps the regional representation to channelwise scale and offset terms,
    each bounded by 0.5 tanh(.), applied as (1 + gamma) * h + beta after each block.
    """

    def __init__(self, in_channels, depthwise, film=False, channels=12, regional_features=16, width=16):
        super().__init__()
        self.channels = channels
        self.projection = nn.Conv1d(in_channels, channels, 1)
        self.blocks = nn.ModuleList(TemporalResidualBlock(channels, 7, d, depthwise) for d in (1, 2, 4))
        self.head_hidden = nn.Linear(regional_features + len(MONTHS) * channels, width)
        self.head_output = nn.Linear(width, 1)
        nn.init.zeros_(self.head_output.weight)
        nn.init.zeros_(self.head_output.bias)
        self.film = None
        self.use_film = film
        if film:
            self.add_film(regional_features)

    def add_film(self, regional_features=16):
        """Attach zero-initialized FiLM maps (the modulation is the identity until they are trained)."""
        self.film = nn.ModuleList(nn.Linear(regional_features, 2 * self.channels) for _ in range(3))
        for layer in self.film:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.film.to(self.projection.weight)
        self.use_film = True

    def encode(self, x, regional):
        z = gelu(self.projection(x.transpose(1, 2)))
        for i, block in enumerate(self.blocks):
            z = block(z)
            if self.use_film and self.film is not None:
                c = 0.5 * torch.tanh(self.film[i](regional))
                gamma, beta = c[:, :self.channels], c[:, self.channels:]
                z = (1 + gamma[:, :, None]) * z + beta[:, :, None]
        return z

    def forward(self, x, regional):
        z = self.encode(x, regional)
        monthly = torch.cat([z[:, :, a:b].mean(-1) for a, b in MONTHS], -1)
        h = gelu(self.head_hidden(torch.cat([regional, monthly], -1)))
        return h, self.head_output(h)[..., 0]


class HistoryNetwork(nn.Module):
    """Dense 34 -> 16 -> 1 network with a GELU hidden layer, applied to each of the three reference years.

    Used twice: as the history weighting network (scores -> tanh -> softmax over the reference years) and as the
    history correction network (mean of the three scalar outputs)."""

    def __init__(self, in_features=34, width=16):
        super().__init__()
        self.hidden = nn.Linear(in_features, width)
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(gelu(self.hidden(x)))[..., 0]


class ResidualMLP(nn.Module):
    """Correction network: dense in -> 64 -> 64 -> 1 with GELU, a skip connection that adds the second hidden layer
    to the first, and a bounded output BOUND * tanh(.). Returns the 64-element hidden representation and the output.

    Used as the prefecture correction network (294 inputs), the municipal correction network (147 inputs), and the
    SST fusion network (64 + 16 inputs)."""

    def __init__(self, in_features, width=64, bound=CORRECTION_BOUND):
        super().__init__()
        self.hidden1 = nn.Linear(in_features, width)
        self.hidden2 = nn.Linear(width, width)
        self.output = nn.Linear(width, 1)
        self.bound = bound
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        h = gelu(self.hidden1(x))
        h = h + gelu(self.hidden2(h))
        return h, self.bound * torch.tanh(self.output(h)[..., 0])


def _pad_sphere(x):
    """Circular padding along longitude and edge-value padding along latitude (x: N, C, lat, lon)."""
    x = F.pad(x, (1, 1, 0, 0), mode="circular")
    return F.pad(x, (0, 0, 1, 1), mode="replicate")


class SSTEncoder(nn.Module):
    """SST encoder: eight 10 x 36 channels (six standardized monthly anomaly maps, ocean fraction, mask indicator)
    -> 16 SST-map features. Two 3x3 convolutions with 8 channels (the second residual), 2x2 and 1x3 average pooling,
    and a dense layer 240 -> 16, all with GELU."""

    def __init__(self, in_channels=8, channels=8, features=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3)
        self.conv2 = nn.Conv2d(channels, channels, 3)
        self.dense = nn.Linear(5 * 6 * channels, features)

    def forward(self, image):
        a = gelu(self.conv1(_pad_sphere(image)))            # N, 8, 10, 36
        a = F.avg_pool2d(a, 2)                               # N, 8, 5, 18
        a = a + gelu(self.conv2(_pad_sphere(a)))
        a = F.avg_pool2d(a, (1, 3))                          # N, 8, 5, 6
        flat = a.permute(0, 2, 3, 1).reshape(len(a), -1)     # latitude, longitude, channel order
        return gelu(self.dense(flat))


class SSTDecoder(nn.Module):
    """Decoder used only for masked-reconstruction pretraining: dense 16 -> 64 -> 2160, reshaped to 6 x 10 x 36."""

    def __init__(self, features=16, width=64, months=6, shape=(10, 36)):
        super().__init__()
        self.hidden = nn.Linear(features, width)
        self.output = nn.Linear(width, months * shape[0] * shape[1])
        self.shape = (months,) + tuple(shape)

    def forward(self, code):
        return self.output(gelu(self.hidden(code))).reshape((len(code),) + self.shape)
