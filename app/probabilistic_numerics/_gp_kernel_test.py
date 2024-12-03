import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Define symbolic variables
t1 = sp.symbols("t1")
t2 = sp.symbols("t2")
tau = sp.symbols("tau")
bound = sp.symbols("bound")

# State transition matrix
F = sp.Matrix([[0, 1], [0, 0]])

# L matrix
L = sp.Matrix([0, 1])

# Define kernel as integral
kernel = sp.integrate(
    sp.exp(F * (t1 - tau)) @ L @ L.T @ sp.exp(F.T * (t2 - tau)),
    (tau, 0, sp.Min(t1, t2)),
)
k00 = kernel[0, 0]
k10 = kernel[1, 0]
k01 = kernel[0, 1]
k11 = kernel[1, 1]

sp.pprint(kernel)

# sp.pprint(k00)
# sp.pprint(k10)
# sp.pprint(k01)
# sp.pprint(k11)

# print(k00)
# print(k10)
# print(k01)
# print(k11)


funcs = [sp.lambdify((t1, t2), expr, "numpy") for expr in [k00, k01, k01, k11]]
funcs = np.array([[funcs[0], funcs[1]], [funcs[2], funcs[3]]])

print(funcs[1, 1](1, 2))

import sklearn.gaussian_process as gp
import sklearn.gaussian_process.kernels as kernels


class CustomKernel(kernels.Kernel):
    def __init__(self):
        super(CustomKernel, self).__init__()

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        print(X)
        output = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                (x, x_i) = X[i]
                (y, y_i) = Y[j]
                output[i, j] = funcs[int(x_i), int(y_i)](x, y)
                print(x_i, y_i, output[i, j])
        print(output)
        return output

    def diag(self, X):
        return np.diag(self(X))

    def is_stationary(self):
        return False


model = gp.GaussianProcessRegressor(kernel=CustomKernel(), n_restarts_optimizer=10)


n = 50

# Test points
test_x = np.linspace(0, 10, n)

test_x0 = np.stack([test_x, np.zeros(n)], axis=1)
# test_x1 = np.stack([test_x, np.ones(n)], axis=1)

# Example training data
timesteps = np.linspace(0, 10, n).reshape(-1, 1)
observations = np.stack(
    [np.sin(2 * timesteps).squeeze(), np.cos(2 * timesteps).squeeze()], axis=1
)

observations0 = observations[:, 0]
observations1 = observations[:, 1]

tessst = np.array(
    [
        [1, 0],
        [1, 1],
        [2, 0],
        [2, 1],
    ]
)
obsss = np.array([1, 2, 3, 4])

model.fit(tessst, obsss)

# Plotting
for i in range(2):
    plt.figure(figsize=(10, 6))
    # Plot training data
    plt.plot(timesteps.numpy(), observations[:, i].numpy(), "k*", label="Observed Data")
    # # Plot predictive mean
    # plt.plot(test_x.numpy(), mean[:, i].numpy(), 'b', label='Predictive Mean')
    # # Shade in confidence interval
    # plt.fill_between(test_x.squeeze().numpy(),
    #                  lower[:, i].numpy(),
    #                  upper[:, i].numpy(),
    #                  alpha=0.3, label='Confidence Interval')
    # plt.title(f'GP Prediction for Output Dimension {i+1}')
    plt.xlabel("Time")
    plt.ylabel("Observation")
    plt.legend()
    plt.show()
