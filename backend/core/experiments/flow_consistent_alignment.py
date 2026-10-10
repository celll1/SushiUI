"""Experimental affine codec initialization with an exact native-space inverse."""
import math

import torch

from core.models.components.vae_alignment import SpatialLatentBridge


class AffineFlowCalibrator:
    """Fit clean pairs, then constrain A B = I and A b + a = 0."""

    def __init__(self, native_channels, target_channels, *, ridge):
        if (any(type(n) is not int for n in (native_channels, target_channels))
                or not 1 <= native_channels <= target_channels <= 1024):
            raise ValueError("Affine flow calibration requires 1 <= native channels <= target channels <= 1024")
        if type(ridge) not in (int, float) or not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("Calibration ridge must be finite and positive")
        self.native_channels, self.target_channels = native_channels, target_channels
        self.ridge, self.count = ridge, 0
        self.xsum = torch.zeros(target_channels, dtype=torch.float64)
        self.ysum = torch.zeros(native_channels, dtype=torch.float64)
        self.xx = torch.zeros(target_channels, target_channels, dtype=torch.float64)
        self.yy = torch.zeros(native_channels, native_channels, dtype=torch.float64)
        self.xy = torch.zeros(target_channels, native_channels, dtype=torch.float64)

    @staticmethod
    def _rows(value, channels):
        if value.ndim not in (4, 5) or value.shape[1] != channels or not value.numel():
            raise ValueError("Affine calibration requires nonempty BCHW/BCFHW with declared channels")
        value = value.detach().cpu().double().movedim(1, -1).reshape(-1, channels)
        if not torch.isfinite(value).all():
            raise ValueError("Affine calibration contains nonfinite values")
        return value

    @torch.no_grad()
    def add(self, native, target):
        if native.ndim != target.ndim or native.shape[:1]+native.shape[2:] != target.shape[:1]+target.shape[2:]:
            raise ValueError("Affine calibration requires equal batch/frame/spatial grids")
        x, y = self._rows(target, self.target_channels), self._rows(native, self.native_channels)
        self.count += len(x)
        self.xsum.add_(x.sum(0)); self.ysum.add_(y.sum(0))
        self.xx.add_(x.T@x); self.yy.add_(y.T@y); self.xy.add_(x.T@y)

    @torch.no_grad()
    def finish(self, *, orthogonal_input=False):
        if type(orthogonal_input) is not bool:
            raise ValueError("Orthogonal input selection must be boolean")
        if self.count <= self.target_channels:
            raise ValueError("Affine calibration has insufficient observations")
        xm, ym = self.xsum/self.count, self.ysum/self.count
        xx = self.xx/self.count-torch.outer(xm, xm)
        yy = self.yy/self.count-torch.outer(ym, ym)
        xy = self.xy/self.count-torch.outer(xm, ym)
        def solve(gram, cross):
            penalty = self.ridge*gram.diag().mean().clamp_min(1e-12)
            return torch.linalg.solve(gram+penalty*torch.eye(len(gram), dtype=gram.dtype), cross)
        a = solve(xx, xy).T
        a_bias = ym-a@xm
        b = solve(yy, xy.T).T
        b_bias = xm-b@ym
        u, singular, vh = torch.linalg.svd(a, full_matrices=False)
        if not torch.isfinite(singular).all() or singular[-1] <= singular[0]*1e-10:
            raise ValueError("Affine flow calibration requires a full-row-rank input map")
        fitted_condition = float(singular[0]/singular[-1])
        if orthogonal_input:
            a = u@vh
            a_bias = ym-a@xm
            singular = torch.ones_like(singular)
        inverse = (vh.T/singular)@u.T
        b += inverse@(torch.eye(self.native_channels, dtype=a.dtype)-a@b)
        b_bias += inverse@(-a_bias-a@b_bias)
        self.metrics = dict(input_condition_number=float(singular[0]/singular[-1]),
                            fitted_input_condition_number=fitted_condition,
                            orthogonal_input=orthogonal_input,
                            inverse_norm=float(torch.linalg.matrix_norm(inverse)),
                            composition_max_error=float((a@b-torch.eye(self.native_channels)).abs().max()),
                            bias_closure_max_error=float((a@b_bias+a_bias).abs().max()))
        with torch.random.fork_rng(devices=[]):
            incoming = SpatialLatentBridge(self.target_channels, self.native_channels)
            outgoing = SpatialLatentBridge(self.native_channels, self.target_channels)
        for bridge, matrix, bias in ((incoming, a, a_bias), (outgoing, b, b_bias)):
            kernel = torch.zeros_like(bridge.projection.weight)
            kernel[:, :, 1, 1] = matrix.to(kernel)
            bridge.initialize(kernel.flatten(1).T, bias.to(kernel)).requires_grad_(False)
        return incoming, outgoing
