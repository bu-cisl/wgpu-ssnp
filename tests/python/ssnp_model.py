"""SSNP forward model."""
import math
import torch
from torch import Tensor
from torch.nn import Module
from torch.types import Device

def scatter_factor(n: Tensor, res_z: float = 0.1, dz: float = 1, n0: float = 1.33) -> Tensor:
	return (2*torch.pi*res_z / n0)**2 * dz * n * (2*n0 + n)

def diffract(uf: Tensor, ub: Tensor, res: tuple[float, ...] = (0.1, 0.1, 0.1), dz: float = 1) -> tuple[Tensor, Tensor]:

	assert uf.shape == ub.shape, 'uf and ub must have the same shape'

	cgamma = c_gamma(res, uf.shape, device=uf.device)
	kz = 2 * torch.pi * res[0] * cgamma
	eva = torch.exp(torch.clamp((cgamma - 0.2) * 5, max=0))

	p_mat = torch.stack([torch.cos(kz * dz), torch.sin(kz * dz) / kz,
			-torch.sin(kz * dz) * kz, torch.cos(kz * dz)])
	
	p_mat *= eva

	uf_new1 = p_mat[0] * uf
	uf_new2 = p_mat[1] * ub
	uf_new = uf_new1 + uf_new2

	ub_new1 = p_mat[2] * uf
	ub_new2 = p_mat[3] * ub
	ub_new = ub_new1 + ub_new2

	return uf_new, ub_new

def c_gamma(res: tuple[float, ...], shape: tuple[int, ...], device: str | Device = 'cpu', dtype = torch.float32) -> Tensor:
	def _near_0(size):
		return torch.fmod(torch.arange(size, device=device).to(dtype) / size + 0.5, 1) - 0.5

	eps = 1E-8
	c_beta, c_alpha = [_near_0(size) / resolution for size, resolution in zip(shape[-2:], res[-2:])]
	return torch.sqrt(torch.clamp(1 - (torch.square(c_alpha) + torch.square(c_beta[:, None])), min=eps)).unsqueeze(0)

def binary_pupil(shape: tuple[int, ...], na: float, res: tuple[float, ...] = (0.1, 0.1, 0.1), device: str | Device = 'cpu') -> Tensor:
	cgamma = c_gamma(res, shape, device=device)
	mask = torch.greater(cgamma, (1 - na ** 2)**0.5)
	return mask

def tilt(shape: tuple[int, ...], c_ba, res, trunc: bool = True, device: str | Device = 'cpu') -> Tensor:

		norm = torch.tensor(shape) * torch.tensor(res[1:])
		norm = norm.view(1, 2)
		norm = norm.to(device=device)

		if trunc:
			factor = torch.trunc(c_ba * norm).T
		else:
			factor = (c_ba * norm).T

		xr = torch.arange(shape[1], device=device).view(1,1,-1).to(dtype=torch.complex128)
		xr = (2j * torch.pi / shape[1]) * factor[1].reshape(-1,1,1) * xr
		xr.exp_()

		yr = torch.arange(shape[0], device=device).view(1,-1,1).to(dtype=torch.complex128)
		yr = (2j * torch.pi / shape[0]) * factor[0].reshape(-1,1,1) * yr
		yr.exp_()

		out = xr * yr

		# normalize by center point value
		out /= out[:, *(i // 2 for i in shape)].clone().view(-1, 1, 1)
		return out

def merge_prop(uf: Tensor, ub: Tensor, res: tuple[float, ...] = (0.1, 0.1, 0.1)) -> tuple[Tensor, Tensor]:

	assert uf.device == ub.device, 'uf and ub must be on the same device'
	
	kz = c_gamma(res, uf.shape, device=uf.device) * (2 * torch.pi * res[0])

	uf_new = uf + ub
	ub_new = (uf - ub) * 1j * kz
	return uf_new, ub_new

def split_prop(uf: Tensor, ub: Tensor, res: tuple[float, ...] = (0.1, 0.1, 0.1)) -> tuple[Tensor, Tensor]:

	assert uf.device == ub.device, 'uf and ub must be on the same device'

	kz = c_gamma(res, uf.shape, device=uf.device) * (2 * torch.pi * res[0])

	ub_new = (uf + 1j*ub / kz) / 2
	uf_new = uf - ub_new

	return uf_new, ub_new

class SSNPBeam(Module):

	def __init__(
			self,
			res: tuple[float, ...] = (0.1, 0.1, 0.1),
			na: float = 0.65,
			angles: int = 32,
			intensity: bool = True,
		):
		super().__init__()

		self.res = res
		self.na = na
		self.angles = torch.nn.Parameter(torch.zeros(angles, 2), requires_grad=True)
		self.intensity = intensity

	def forward(self, n: Tensor) -> Tensor:

		shape = n.shape[-2:]
		
		# configure input feild
		Forward = torch.fft.fft2(tilt(shape, c_ba=self.angles, res=self.res, device=n.device))
		Backward = torch.zeros_like(Forward)
		U, UD = merge_prop(Forward, Backward, res=self.res)

	
		# propagate the wave through the RI distribution
		for slice in n:

			# propagate the wave 1.0*Δz
			U, UD = diffract(U, UD, res=self.res, dz=1.0)

			# convert feild to spatial domain
			u = torch.fft.ifft2(U)

			# compute scattering affects
			UD = UD - torch.fft.fft2(scatter_factor(slice, res_z=self.res[0], dz=1, n0=1.33) * u)

		# propagate the wave back to the focal plane
		U, UD = diffract(U, UD, res=self.res, dz = -len(n)/2)

		# merge the forward and backward feilds from u and ∂u
		Forward, _ = split_prop(U, UD, res=self.res)
		Forward = Forward * binary_pupil(shape, self.na, res=self.res, device=n.device)

		# return the intensity
		result = torch.abs(torch.fft.ifft2(Forward))
		return result**2 if self.intensity else result
