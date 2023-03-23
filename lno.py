import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import laplace, invlaplace


class LNO2D(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
    ):

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_heads = num_heads
        self.block_size = self.hidden_size // self.num_heads
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                self.num_heads,
                self.block_size,
                self.block_size * self.hidden_size_factor,
            )
        )
        self.b1 = nn.Parameter(
            self.scale
            * torch.randn(2, self.num_heads, self.block_size * self.hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                self.num_heads,
                self.block_size * self.hidden_size_factor,
                self.block_size,
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_heads, self.block_size)
        )

    def laplace_transform(self, x):
        x_np = x.detach().cpu().numpy()
        result = laplace(laplace(x_np, axis=1), axis=2)
        return torch.from_numpy(result).to(x.device)

    def inverse_laplace_transform(self, x):
        x_np = x.detach().cpu().numpy()
        result = invlaplace(invlaplace(x_np, axis=1), axis=2)
        return torch.from_numpy(result).to(x.device)

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = self.laplace_transform(x)
        x = x.reshape(B, H, W // 2 + 1, self.num_heads, self.block_size)

        o1_real = torch.zeros(
            [
                B,
                H,
                W // 2 + 1,
                self.num_heads,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o1_imag = torch.zeros(
            [
                B,
                H,
                W // 2 + 1,
                self.num_heads,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
        ] = F.relu(
            torch.einsum(
                "...bi,bio->...bo",
                x[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ].real,
                self.w1[0],
            )
            - torch.einsum(
                "...bi,bio->...bo",
                x[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ].imag,
                self.w1[1],
            )
            + self.b1[0]
        )

        o1_imag[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
        ] = F.relu(
            torch.einsum(
                "...bi,bio->...bo",
                x[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ].imag,
                self.w1[0],
            )
            + torch.einsum(
                "...bi,bio->...bo",
                x[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ].real,
                self.w1[1],
            )
            + self.b1[1]
        )

        o2_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            torch.einsum(
                "...bi,bio->...bo",
                o1_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[0],
            )
            - torch.einsum(
                "...bi,bio->...bo",
                o1_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[1],
            )
            + self.b2[0]
        )

        o2_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            torch.einsum(
                "...bi,bio->...bo",
                o1_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[0],
            )
            + torch.einsum(
                "...bi,bio->...bo",
                o1_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[1],
            )
            + self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = self.inverse_laplace_transform(x)
        x = x.type(dtype)

        return x + bias


def test_lno2d():
    batch_size = 4
    height = 32
    width = 32
    channels = 3

    lno2d = LNO2D(hidden_size=64)
    input_tensor = torch.randn(batch_size, height, width, channels)
    output_tensor = lno2d(input_tensor)
    assert input_tensor.shape == output_tensor.shape, f"Output shape {output_tensor.shape} doesn't match input shape {input_tensor.shape}"