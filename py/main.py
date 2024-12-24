import numpy as np
import ssnp_model

def test_scatter_factor():
    n = np.array([1.0, 2.0, 3.0])

    python_result = ssnp_model.scatter_factor(n)

    print("Python scatter_factor result:")
    print(python_result)

if __name__ == "__main__":
    test_scatter_factor()
