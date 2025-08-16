"""BPM Forward Model"""

import logging
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.types import Device


# --------------------
# Ops
# --------------------

@torch.no_grad()
def c_gamma(
    res: tuple[float, ...],
    shape: tuple[int, ...],
    device: str | Device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    def _near_0(size):
        return (
            torch.fmod(torch.arange(size, device=device).to(dtype) / size + 0.5, 1)
            - 0.5
        )

    eps = 1e-8
    c_beta, c_alpha = [
        _near_0(size) / resolution for size, resolution in zip(shape[-2:], res[-2:])
    ]
    return torch.sqrt(
        torch.clamp(
            1 - (torch.square(c_alpha) + torch.square(c_beta[:, None])), min=eps
        )
    ).unsqueeze(0)


@torch.no_grad()
def tilt(
    c_ba: Tensor,
    shape: tuple[int, ...],
    device: str | Device = "cpu",
    res=(0.1, 0.1, 0.1)
) -> Tensor:
    """Return Fourier-domain delta for given tilt angles."""
    assert len(shape) == 2

    N = np.prod(shape)

    x_shifts = [
        int(torch.round(ca * res[2] * shape[1]))
        for _, ca in c_ba
    ]

    y_shifts = [
        int(torch.round(cb * res[1] * shape[0]))
        for cb, _ in c_ba
    ]

    angles = torch.arange(len(c_ba))

    fields = torch.zeros(len(c_ba), *shape, device=device, dtype=torch.complex64)

    fields[angles, y_shifts, x_shifts] = N

    return fields


def binary_pupil(
    Field: Tensor,
    na: float,
    res: tuple[float, ...] = (0.1, 0.1, 0.1)
) -> Tensor:
    cgamma = c_gamma(res, Field.shape[-2:], device=Field.device)
    mask = torch.greater(cgamma, (1 - na**2) ** 0.5)
    return Field * mask


# --------------------
# BPM physics
# --------------------

def scatter(
    field: Tensor, n: Tensor, res_z: float = 0.1, dz: float = 1, n0: float = 1.33
) -> Tensor:
    """Apply scattering in SPATIAL domain, return updated FOURIER domain field."""
    return torch.fft.fft2(
        torch.fft.ifft2(field) * torch.exp(1j * (2 * torch.pi * res_z / n0) * dz * n)
    )


def diffract(
    field: Tensor, res: tuple[float, ...] = (0.1, 0.1, 0.1), dz: float = 1
) -> Tensor:
    """Propagate a field in FOURIER domain."""
    match field.dtype:
        case torch.complex64:
            cg = c_gamma(res, field.shape, device=field.device, dtype=torch.float32)
        case torch.complex128:
            cg = c_gamma(res, field.shape, device=field.device, dtype=torch.float64)
        case _:
            raise ValueError("field must be complex tensors")

    kz = 2 * torch.pi * res[0] * cg
    eva = torch.exp(torch.clamp((cg - 0.2) * 5, max=0))

    return field * torch.exp(1j * kz * eva * dz)


# --------------------
# Beam classes
# --------------------

class _Beam(Module):
    def __init__(
        self,
        c_ba: Tensor | None = None,
        res: tuple[float, ...] = (0.1, 0.1, 0.1),
        n0: float = 1.33,
        na: float = 0.65,
        intensity: bool = True,
        ad_impl: str = "naive",
        angles: int = 32,
    ) -> None:
        super().__init__()
        self.res = res
        self.na = na
        self.n0 = n0
        self.intensity = intensity
        self.ad_impl = ad_impl

        if c_ba is not None:
            cb = torch.as_tensor(c_ba, dtype=torch.float32)
            if cb.ndim == 1:
                cb = cb.view(1, 2)
        else:
            cb = torch.zeros(angles, 2, dtype=torch.float32)

        self.c_ba = Parameter(cb.clone(), requires_grad=False)

    def forward(self, n: Tensor, focal_offset: float | None = None) -> Tensor:
        Field = self.compute_field(n)

        if focal_offset is not None:
            Field = diffract(Field, self.res, dz=-focal_offset)

        Field = binary_pupil(Field, na=self.na, res=self.res)

        field = torch.fft.ifft2(Field)

        return field.abs().pow(2) if self.intensity else field.abs()

    def compute_field(self, n: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement this method.")

    def compute_sqrt_indices(self, n: Tensor) -> list[int]:
        save_index = int(np.ceil(np.sqrt(len(n))))
        save_length = save_index

        save_indices = [0, save_index]

        save_length -= 1
        save_index += save_length

        while save_index < (len(n) - 2):
            save_indices.append(save_index)
            save_length -= 1
            save_index += save_length

        save_indices.append(-1)
        return save_indices


class BPMBeam(_Beam):
    """Wrapper for Beam Propagation Method."""

    def compute_field(self, n: Tensor) -> Tensor:
        """Compute the scalar field at the focal plane using BPM."""

        def propagate(n_vol: Tensor, Field: Tensor) -> Tensor:
            for slice in n_vol:
                Field = diffract(Field, res=self.res, dz=1.0)
                Field = scatter(Field, slice, self.res[0], dz=1.0, n0=self.n0)
            return Field

        shape = n.shape[-2:]

        # ops.tilt returns Fourier-domain delta
        Field = tilt(self.c_ba, shape, device=n.device, res=self.res)

        match self.ad_impl:
            case "sqrt":
                save_indices = self.compute_sqrt_indices(n)
                for start, end in zip(save_indices[:-1], save_indices[1:]):
                    Field = torch.utils.checkpoint.checkpoint(
                        propagate(n[start:end], Field)
                    )
            case "naive":
                Field = propagate(n, Field)
            case _:
                logging.warning(
                    f"Unknown ad_impl {self.ad_impl}, using naive implementation"
                )
                Field = propagate(n, Field)

        Field = diffract(Field, res=self.res, dz=-len(n) / 2)

        return Field
