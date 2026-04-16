"""Born forward model."""

import torch
from torch import Tensor
from torch.nn import Module, Parameter


@torch.no_grad()
def c_gamma(
    res: tuple[float, ...],
    shape: tuple[int, ...],
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    def _near_0(size: int) -> Tensor:
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


def diffract(
    field: Tensor, res: tuple[float, ...] = (0.1, 0.1, 0.1), dz: float = 1
) -> Tensor:
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


def binary_pupil(
    field: Tensor,
    na: float,
    res: tuple[float, ...] = (0.1, 0.1, 0.1),
) -> Tensor:
    cg = c_gamma(res, field.shape[-2:], device=field.device)
    mask = torch.greater(cg, (1 - na**2) ** 0.5)
    return field * mask


def born_propagation(
    delta_n: Tensor,
    c_ba: Tensor,
    res: tuple[float, ...] = (0.1, 0.1, 0.1),
    n0: float = 1.33,
) -> Tensor:
    shifts = [
        (
            int(torch.round(c_ba_i[1] * res[2] * delta_n.shape[2])),
            int(torch.round(c_ba_i[0] * res[1] * delta_n.shape[1])),
        )
        for c_ba_i in c_ba
    ]

    field = torch.fft.fft2(
        torch.ones(delta_n.shape[1:], dtype=torch.float32, device=delta_n.device)
    )

    fields = torch.stack(
        [
            torch.roll(
                field,
                shifts=shift,
                dims=(1, 0),
            )
            for shift in shifts
        ],
        dim=0,
    )

    scatter_potential = torch.fft.fft2(
        (2 * torch.pi * res[0] / n0) ** 2 * delta_n * (2 * n0 + delta_n)
    )

    kz = c_gamma(
        res=res, shape=delta_n.shape, device=delta_n.device, dtype=delta_n.dtype
    ).float() * (2 * torch.pi * res[0])

    kz_in = torch.sqrt(1 - c_ba.pow(2).sum(1)) * (2 * torch.pi * res[0])
    defo_kernel = kz - kz_in.view(-1, 1, 1)

    for slice_i, depth in zip(
        scatter_potential, delta_n.shape[0] / 2 - torch.arange(delta_n.shape[0])
    ):
        slice_shift = torch.stack(
            [
                torch.roll(
                    slice_i,
                    shifts=shift,
                    dims=(1, 0),
                )
                for shift in shifts
            ],
            dim=0,
        )

        fields += 1j / 2 * slice_shift * torch.exp(1j * depth * defo_kernel) / kz

    return fields


class BornBeam(Module):
    def __init__(
        self,
        c_ba: Tensor | None = None,
        res: tuple[float, ...] = (0.1, 0.1, 0.1),
        n0: float = 1.33,
        na: float = 0.65,
        intensity: bool = True,
        angles: int = 32,
    ) -> None:
        super().__init__()
        self.res = res
        self.n0 = n0
        self.na = na
        self.intensity = intensity

        if c_ba is not None:
            cb = torch.as_tensor(c_ba, dtype=torch.float32)
            if cb.ndim == 1:
                cb = cb.view(1, 2)
        else:
            cb = torch.zeros(angles, 2, dtype=torch.float32)

        self.c_ba = Parameter(cb.clone(), requires_grad=False)

    def forward(self, n: Tensor, focal_offset: float | None = None) -> Tensor:
        field = born_propagation(n, self.c_ba, self.res, self.n0)

        if focal_offset is not None:
            field = diffract(field, self.res, dz=-focal_offset)

        field = binary_pupil(field, na=self.na, res=self.res)
        field = torch.fft.ifft2(field)

        return field.abs().pow(2) if self.intensity else field.abs()
