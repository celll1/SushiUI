"""Experimental affine codec initialization with an exact native-space inverse."""
import math

import torch
from torch.nn import functional as F

from core.models.components.vae_alignment import SpatialLatentBridge


class AffineFlowCalibrator:
    """Fit clean pairs, then constrain A B = I and A b + a = 0."""

    def __init__(self, native_channels, target_channels, *, ridge, output_kernel_size=1):
        if (any(type(n) is not int for n in (native_channels, target_channels))
                or not 1 <= native_channels <= target_channels <= 1024):
            raise ValueError("Affine flow calibration requires 1 <= native channels <= target channels <= 1024")
        if type(ridge) not in (int, float) or not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("Calibration ridge must be finite and positive")
        if type(output_kernel_size) is not int or output_kernel_size not in (1, 3):
            raise ValueError("Output kernel size must be 1 or 3")
        self.native_channels, self.target_channels = native_channels, target_channels
        self.output_kernel_size = output_kernel_size
        features = native_channels*output_kernel_size**2
        self.ridge, self.count = ridge, 0
        self.xsum = torch.zeros(target_channels, dtype=torch.float64)
        self.ysum = torch.zeros(native_channels, dtype=torch.float64)
        self.xx = torch.zeros(target_channels, target_channels, dtype=torch.float64)
        self.xy = torch.zeros(target_channels, native_channels, dtype=torch.float64)
        self.feature_sum = torch.zeros(features, dtype=torch.float64)
        self.feature_gram = torch.zeros(features, features, dtype=torch.float64)
        self.feature_cross = torch.zeros(features, target_channels, dtype=torch.float64)

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
        if self.output_kernel_size == 1:
            features = y
        else:
            images = native.detach().cpu().double()
            if images.ndim == 5:
                images = images.permute(0, 2, 1, 3, 4).reshape(-1, self.native_channels, *images.shape[-2:])
            patches = F.unfold(images, kernel_size=3, padding=1)
            features = patches.transpose(1, 2).reshape(-1, self.native_channels*9)
        self.count += len(x)
        self.xsum.add_(x.sum(0)); self.ysum.add_(y.sum(0))
        self.xx.add_(x.T@x); self.xy.add_(x.T@y)
        self.feature_sum.add_(features.sum(0))
        self.feature_gram.add_(features.T@features)
        self.feature_cross.add_(features.T@x)

    @torch.no_grad()
    def finish(self, *, orthogonal_input=False, match_native_covariance=False):
        if type(orthogonal_input) is not bool:
            raise ValueError("Orthogonal input selection must be boolean")
        if type(match_native_covariance) is not bool:
            raise ValueError("Native covariance matching must be boolean")
        if match_native_covariance and not orthogonal_input:
            raise ValueError("Native covariance matching requires orthogonal input selection")
        if self.count <= self.target_channels:
            raise ValueError("Affine calibration has insufficient observations")
        xm, ym = self.xsum/self.count, self.ysum/self.count
        xx = self.xx/self.count-torch.outer(xm, xm)
        xy = self.xy/self.count-torch.outer(xm, ym)
        fm = self.feature_sum/self.count
        gram = self.feature_gram/self.count-torch.outer(fm, fm)
        cross = self.feature_cross/self.count-torch.outer(fm, xm)
        def solve(gram, cross):
            penalty = self.ridge*gram.diag().mean().clamp_min(1e-12)
            return torch.linalg.solve(gram+penalty*torch.eye(len(gram), dtype=gram.dtype), cross)
        a = solve(xx, xy).T
        a_bias = ym-a@xm
        b = solve(gram, cross).T
        b_bias = xm-b@fm
        u, singular, vh = torch.linalg.svd(a, full_matrices=False)
        if not torch.isfinite(singular).all() or singular[-1] <= singular[0]*1e-10:
            raise ValueError("Affine flow calibration requires a full-row-rank input map")
        fitted_condition = float(singular[0]/singular[-1])
        if orthogonal_input:
            a = u@vh
        center = torch.arange(self.native_channels)*self.output_kernel_size**2+self.output_kernel_size**2//2
        native_covariance = gram[center][:, center]
        if match_native_covariance:
            def covariance_power(value, power):
                values, vectors = torch.linalg.eigh((value+value.T)/2)
                if values[-1] <= 0 or values[0] <= values[-1]*1e-10:
                    raise ValueError("Native covariance matching requires positive definite covariance")
                return (vectors*values.pow(power))@vectors.T
            projected = a@xx@a.T
            a = covariance_power(native_covariance, .5)@covariance_power(projected, -.5)@a
        if orthogonal_input:
            a_bias = ym-a@xm
            u, singular, vh = torch.linalg.svd(a, full_matrices=False)
        inverse = (vh.T/singular)@u.T
        kernels = self.output_kernel_size**2
        b = b.reshape(self.target_channels, self.native_channels, kernels)
        identity = torch.zeros(self.native_channels, self.native_channels, kernels, dtype=a.dtype)
        identity[:, :, kernels//2] = torch.eye(self.native_channels, dtype=a.dtype)
        composition = torch.einsum('ij,jkl->ikl', a, b)
        b += torch.einsum('ij,jkl->ikl', inverse, identity-composition)
        b_bias += inverse@(-a_bias-a@b_bias)
        self.metrics = dict(input_condition_number=float(singular[0]/singular[-1]),
                            fitted_input_condition_number=fitted_condition,
                            orthogonal_input=orthogonal_input,
                            match_native_covariance=match_native_covariance,
                            covariance_max_error=float((a@xx@a.T-native_covariance).abs().max()),
                            output_kernel_size=self.output_kernel_size,
                            inverse_norm=float(torch.linalg.matrix_norm(inverse)),
                            composition_max_error=float((torch.einsum('ij,jkl->ikl', a, b)-identity).abs().max()),
                            bias_closure_max_error=float((a@b_bias+a_bias).abs().max()))
        with torch.random.fork_rng(devices=[]):
            incoming = SpatialLatentBridge(self.target_channels, self.native_channels)
            outgoing = SpatialLatentBridge(self.native_channels, self.target_channels)
        for bridge, matrix, bias in ((incoming, a, a_bias), (outgoing, b, b_bias)):
            kernel = torch.zeros_like(bridge.projection.weight)
            if matrix.ndim == 2 or self.output_kernel_size == 1:
                kernel[:, :, 1, 1] = matrix.reshape(bridge.projection.out_channels, bridge.projection.in_channels).to(kernel)
            else:
                kernel.copy_(matrix.reshape_as(kernel))
            bridge.initialize(kernel.flatten(1).T, bias.to(kernel)).requires_grad_(False)
        return incoming, outgoing
