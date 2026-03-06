from torch import nn


class WaveletNoise(nn.Module):
    def __init__(
        self,
        levels=5,
        domain=10.0,
        base_sigma=0.9,
        hurst=0.8,
        mountain_bias=0.6,
    ):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be >= 1")

        self.levels = levels
        self.domain = float(domain)
        self.base_sigma = float(base_sigma)
        self.hurst = float(hurst)
        self.mountain_bias = float(mountain_bias)

        coarse_size = max(2, 2 ** (levels - 3))
        self.scaling_coeffs = nn.Parameter(torch.empty(coarse_size, coarse_size))

        coarse_axis = torch.linspace(-1.0, 1.0, coarse_size)
        coarse_x, coarse_y = torch.meshgrid(coarse_axis, coarse_axis, indexing="ij")
        self.register_buffer("_scaling_cx", coarse_x)
        self.register_buffer("_scaling_cy", coarse_y)

        self.detail_coeffs = nn.ParameterList()
        for level in range(levels):
            size = 2**level
            coeff = nn.Parameter(torch.empty(size, size))
            self.detail_coeffs.append(coeff)

            axis = torch.linspace(-1.0, 1.0, size)
            cx, cy = torch.meshgrid(axis, axis, indexing="ij")
            self.register_buffer(f"_detail_cx_{level}", cx)
            self.register_buffer(f"_detail_cy_{level}", cy)

        self.reset_parameters()

    @staticmethod
    def _radial_taper(cx, cy, sharpness):
        radius2 = cx.square() + cy.square()
        return torch.exp(-sharpness * radius2)

    def reset_parameters(self):
        with torch.no_grad():
            self.scaling_coeffs.normal_(mean=1.1, std=0.35)
            coarse_taper = self._radial_taper(
                self._scaling_cx, self._scaling_cy, sharpness=1.8
            )
            self.scaling_coeffs.mul_(0.45 + 0.9 * coarse_taper)

            for level, coeff in enumerate(self.detail_coeffs):
                amp = 2.0 ** (-self.hurst * level)
                coeff.normal_(
                    mean=0.22 * self.mountain_bias * amp,
                    std=0.55 * amp,
                )
                cx = getattr(self, f"_detail_cx_{level}")
                cy = getattr(self, f"_detail_cy_{level}")
                taper = self._radial_taper(cx, cy, sharpness=0.9 + 0.2 * level)
                coeff.mul_(0.3 + 0.9 * taper)

    def _ricker_sum(self, xy, coeffs, cx, cy, sigma):
        sigma2 = float(sigma) ** 2 + 1e-8
        dx = xy[:, None, None, 0] - cx[None, :, :]
        dy = xy[:, None, None, 1] - cy[None, :, :]
        r2 = (dx.square() + dy.square()) / sigma2

        # 2D Mexican-hat (Ricker) wavelet basis.
        basis = (2.0 - r2) * torch.exp(-0.5 * r2)
        return torch.einsum("bij,ij->b", basis, coeffs)

    def _scaling_sum(self, xy):
        dx = xy[:, None, None, 0] - self._scaling_cx[None, :, :]
        dy = xy[:, None, None, 1] - self._scaling_cy[None, :, :]
        r2 = (dx.square() + dy.square()) / (self.base_sigma * 1.6) ** 2
        basis = torch.exp(-0.5 * r2)
        return torch.einsum("bij,ij->b", basis, self.scaling_coeffs)

    def forward(self, coords):
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must have shape [N, 2]")

        xy = coords / self.domain
        out = self._scaling_sum(xy)
        for level, coeff in enumerate(self.detail_coeffs):
            sigma = self.base_sigma / (2**level)
            cx = getattr(self, f"_detail_cx_{level}")
            cy = getattr(self, f"_detail_cy_{level}")
            out = out + self._ricker_sum(xy, coeff, cx, cy, sigma)

        # Positive terrain elevation bias while keeping raw coeffs trainable.
        return torch.nn.functional.softplus(out)

    def raw_wavelet_coeffs(self):
        return [self.scaling_coeffs, *self.detail_coeffs]
