import math
import torch
from torch import Tensor

def tilt(shape: tuple[int], angles: Tensor, NA: float= 0.65, res: tuple[float] = (0.1, 0.1, 0.1), trunc: bool = True, device: str = 'cpu') -> Tensor:
	c_ba = NA*torch.stack(
		(
			torch.sin(angles),
			torch.cos(angles)
		),
		dim=1
	)
	# print(f"c_ba: {c_ba.shape} \n{c_ba}")

	norm = torch.tensor(shape) * torch.tensor(res[1:])
	norm = norm.view(1, 2)
	# print(f"norm (shape * resolution): \n{norm}")

	if trunc:
		factor = torch.trunc(c_ba * norm).T
	else:
		factor = (c_ba * norm).T
	# print(f"factor (after truncation check): {factor.shape} \n{factor}")

	xr = torch.arange(shape[1], device=device).view(1,1,-1).to(dtype=torch.complex128)
	xr = (2j * torch.pi / shape[1]) * factor[1].reshape(-1,1,1) * xr
	xr.exp_()
	# print(f"xr (exponential in x direction): \n{xr}")

	yr = torch.arange(shape[0], device=device).view(1,-1,1).to(dtype=torch.complex128)
	yr = (2j * torch.pi / shape[0]) * factor[0].reshape(-1,1,1) * yr
	yr.exp_()
	# print(f"yr (exponential in y direction): \n{yr}")

	out = xr * yr
	# print(f"out: \n{out}")

	# normalize by center point value
	out /= out[:, *(i // 2 for i in shape)].clone()
	return out

def tilt2(shape: tuple[int], angles: Tensor, NA: float= 0.65, res: tuple[float] = (0.1, 0.1, 0.1), trunc: bool = True, device: str = 'cpu') -> Tensor:
	sin_component = torch.sin(angles)
	cos_component = torch.cos(angles)
	print(f"sin: {sin_component}, {sin_component.shape}")
	print(f"cos: {cos_component}, {cos_component.shape}")
	
	c_ba = NA * torch.stack((sin_component, cos_component), dim=1)
	print(f"c_ba: {c_ba.shape} \n{c_ba}")
	
	shape_tensor = torch.tensor(shape)
	res_tensor = torch.tensor(res[1:])
	norm = shape_tensor * res_tensor
	norm = norm.view(1, 2)
	print(f"norm: {norm}")
	
	scaled_c_ba = c_ba * norm
	print(f"scaled_c_ba: {scaled_c_ba}")
	if trunc:
		factor = torch.trunc(scaled_c_ba).T
	else:
		factor = scaled_c_ba.T
	print(f"factor: {factor.shape} \n{factor}")
	return factor
	"""
	x_indices = torch.arange(shape[1], device=device).view(1, 1, -1).to(dtype=torch.complex128)
	factor_x = factor[1].reshape(-1, 1, 1)
	exponent_x = (2j * torch.pi / shape[1]) * factor_x * x_indices
	xr = exponent_x.exp()
	
	y_indices = torch.arange(shape[0], device=device).view(1, -1, 1).to(dtype=torch.complex128)
	factor_y = factor[0].reshape(-1, 1, 1)
	exponent_y = (2j * torch.pi / shape[0]) * factor_y * y_indices
	yr = exponent_y.exp()
	
	out = xr * yr
	
	normalization_indices = tuple(i // 2 for i in shape)
	normalization_values = out[:, normalization_indices].clone()
	out /= normalization_values

	return out
	"""