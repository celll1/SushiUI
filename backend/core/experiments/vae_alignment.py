"""CPU affine probes for paired, normalized VAE latents."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def approved_image_paths(records):
    """Fail closed before any image read: this experiment permits rating general only."""
    paths = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each image requires explicit general-rating metadata")
        tags = {str(tag).strip().casefold() for tag in record.get("tags", [])}
        if (record.get("rating") != "general" or "general" not in tags
                or tags & {"sensitive", "questionable", "explicit"}):
            raise ValueError("Only unambiguous rating-general images are permitted")
        paths.append(record["path"])
    return paths


def rows(latent):
    if latent.ndim != 4:
        raise ValueError("Expected BCHW latents")
    return latent.permute(0, 2, 3, 1).reshape(-1, latent.shape[1])


def pack(latent, factor=2):
    if latent.ndim != 4 or any(n % factor for n in latent.shape[-2:]):
        raise ValueError("Spatial dimensions must be divisible by packing factor")
    return F.pixel_unshuffle(latent, factor)


def unpack(latent, factor=2):
    return F.pixel_shuffle(latent, factor)


@dataclass
class AffineMap:
    weight: torch.Tensor  # row vectors: output = input @ weight + bias
    bias: torch.Tensor

    def __call__(self, latent):
        values = rows(latent).to(self.weight) @ self.weight + self.bias
        return values.reshape(latent.shape[0], *latent.shape[-2:], -1).permute(0, 3, 1, 2).float()

    def then(self, other):
        return AffineMap(self.weight @ other.weight,
                         self.bias @ other.weight + other.bias)

    def flow_state(self, latent, sigma):
        """Map (1-sigma)*clean + sigma*noise; affine bias belongs to clean only."""
        sigma = torch.as_tensor(sigma, dtype=torch.float32)
        if sigma.ndim == 1:
            sigma = sigma.reshape(-1, 1, 1, 1)
        mapped = self(latent)
        return mapped - sigma * self.bias.float().reshape(1, -1, 1, 1)

    def diagnostics(self):
        covariance = self.weight.T @ self.weight
        identity = torch.eye(covariance.shape[0], dtype=covariance.dtype)
        return {
            "rank": int(torch.linalg.matrix_rank(self.weight)),
            "noise_rms_gain": float(covariance.diag().mean().sqrt()),
            "noise_covariance_relative_error": float((covariance - identity).norm() / identity.norm()),
        }


def fit_affine(source, target, ridge=1e-3):
    x, y = rows(source).double(), rows(target).double()
    if source.shape[0:1] + source.shape[2:] != target.shape[0:1] + target.shape[2:]:
        raise ValueError("Paired latent grids must match")
    return _fit_rows(x, y, ridge)


def _fit_rows(x, y, ridge):
    xm, ym = x.mean(0), y.mean(0)
    x, y = x - xm, y - ym
    gram = x.T @ x / len(x)
    penalty = ridge * gram.diag().mean().clamp_min(1e-12)
    weight = torch.linalg.solve(gram + penalty * torch.eye(x.shape[1], dtype=x.dtype), x.T @ y / len(x))
    return AffineMap(weight, ym - xm @ weight)


@dataclass
class SpatialMap(AffineMap):
    kernel_size: int = 3

    def __call__(self, latent):
        features = F.unfold(latent.double(), self.kernel_size, padding=self.kernel_size // 2)
        values = features.transpose(1, 2) @ self.weight + self.bias
        return values.transpose(1, 2).reshape(latent.shape[0], -1, *latent.shape[-2:]).float()


def fit_spatial(source, target, ridge=1e-2):
    if source.shape[0:1] + source.shape[2:] != target.shape[0:1] + target.shape[2:]:
        raise ValueError("Paired latent grids must match")
    features = F.unfold(source.double(), 3, padding=1).transpose(1, 2).reshape(-1, source.shape[1] * 9)
    result = _fit_rows(features, rows(target).double(), ridge)
    return SpatialMap(result.weight, result.bias)


def fit_orthogonal(source, target):
    x, y = rows(source).double(), rows(target).double()
    if x.shape != y.shape:
        raise ValueError("Orthogonal alignment requires equal paired dimensions")
    xm, ym = x.mean(0), y.mean(0)
    u, _, vh = torch.linalg.svd((x - xm).T @ (y - ym))
    weight = u @ vh
    return AffineMap(weight, ym - xm @ weight)


def metrics(prediction, target):
    error = (prediction.float() - target.float()).square().mean()
    energy = target.float().square().mean().clamp_min(1e-12)
    variance = (target.float() - target.float().mean((0, 2, 3), keepdim=True)).square().mean().clamp_min(1e-12)
    return {"mse": float(error), "relative_rmse": float((error / energy).sqrt()),
            "explained_variance": float(1 - error / variance)}


def fold_input(linear, alignment):
    """Compose a new-to-old feature map into an old input Linear."""
    weight = linear.weight.detach().double() @ alignment.weight.T
    old_bias = (linear.bias.detach().double() if linear.bias is not None
                else torch.zeros(linear.out_features, dtype=torch.float64))
    bias = old_bias + alignment.bias @ linear.weight.detach().double().T
    return weight, bias


def ming_patch_order(alignment):
    """pixel_unshuffle is channel-major; Ming patch-2 input is position-major."""
    if alignment.weight.shape[1] != 64:
        raise ValueError("Expected a packed 16-channel Ming patch")
    permutation = torch.arange(64).reshape(16, 4).T.reshape(-1)
    return AffineMap(alignment.weight[:, permutation], alignment.bias[permutation])
