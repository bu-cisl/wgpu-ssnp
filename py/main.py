import numpy as np
import ssnp_model

def test_scatter_factor():
    n = np.array([1.0, 2.0, 3.0])
    python_result = ssnp_model.scatter_factor(n)

    print("Python scatter_factor result:")
    print(python_result)

def test_c_gamma():
    res = (1, 1, 1)
    shape = (3, 3)
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

    res = (1, 1, 1)
    dz = 1
    uf_new, ub_new = ssnp_model.diffract(uf, ub, res, dz)
    print("Python diffract result (uf_new):")
    print(uf_new)
    print("Python diffract result (ub_new):")
    print(ub_new)

if __name__ == "__main__":
    # test_scatter_factor()
    # test_c_gamma()
    test_diffract()