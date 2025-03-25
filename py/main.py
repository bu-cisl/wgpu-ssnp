import numpy as np
import torch
import ssnp_model

def test_scatter_factor():
    n = np.array([5, 21, 65])
    python_result = ssnp_model.scatter_factor(n)

    print("Python scatter_factor result:")
    print(python_result)

def test_c_gamma():
    res = (5.2, 2.2)
    shape = (3, 2)
    gamma_result = ssnp_model.c_gamma(res, shape)

    print("Python c_gamma result:")
    print(gamma_result)

def test_diffract():
    uf = np.array([[1.0 + 0j, 2.0 + 0j, 3.0 + 0j], 
                   [4.0 + 0j, 5.0 + 0j, 6.0 + 0j], 
                   [7.0 + 0j, 8.0 + 0j, 9.0 + 0j]])
    ub = np.array([[9.0 + 0j, 8.0 + 0j, 7.0 + 0j], 
                   [6.0 + 0j, 5.0 + 0j, 4.0 + 0j], 
                   [3.0 + 0j, 2.0 + 0j, 1.0 + 0j]])
    res = (.1, .1, .1)
    dz = 1
    uf_new, ub_new = ssnp_model.diffract(uf, ub, res, dz)

    print("Python diffract result (uf_new):")
    print(uf_new)
    print("Python diffract result (ub_new):")
    print(ub_new)

def test_binary_pupil():
    shape = (5, 4)
    na = 0.7
    res = (0.1, 0.1, 0.1)
    mask = ssnp_model.binary_pupil(shape, na, res)

    print("Python binary_pupil result:")
    print(mask)

def test_tilt():
    shape = (2, 2)
    angles = torch.tensor([50, 100], dtype=torch.float32)
    NA = 0.65
    res = (0.1, 0.1, 0.1)
    trunc = False
    tilt_result = ssnp_model.tilt(shape, angles, NA, res, trunc)

    print("Python tilt result:")
    print(tilt_result)

if __name__ == "__main__":
    test_scatter_factor()
    test_c_gamma()
    test_diffract()
    test_binary_pupil()
    test_tilt()
