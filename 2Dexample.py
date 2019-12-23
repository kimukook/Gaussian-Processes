import numpy as np
import GP

if __name__ == "__main__":
    n = 2
    MS = 50
    lb = -5 * np.ones((n, 1))
    ub = 5 * np.ones((n, 1))

    # Test 1: sin function
    # y = np.sin(x)[0]

    # Test 2: schwefel function
    func = lambda x: - sum(np.multiply(500 * x, np.sin(np.sqrt(abs(500 * x))))) / 250

    gp = GP.GaussianProcessRegression(n, func, 'SE', MS, lb, ub)
    gp.optimizer()

    # posterior plot
    gp.visualizer_2D()

