import time
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from numpy.random import randn

start = time.time()


A = np.array([[1.0, 1.0, 0.5], [0, 1.0, 1.0], [0, 0.0, 1]])
"""State evolution"""
Q = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1.0]])
"""State noise covariance"""

H = np.array([[1.0, 0.0, 0.0]])
"""Observation model"""
R = np.array([[0.1]])
"""Observation model noise covariance"""

initial_obs = np.array([0.0, 0.0, 0.0])

filter = KalmanFilter(dim_x=3, dim_z=1)
filter.x = initial_obs
filter.F = A
filter.Q = Q
filter.H = H
filter.R = R


timesteps = np.arange(0, 500000)
observations = np.cos(timesteps * 0.5) + 1


means = []
covs = []
for i in range(len(timesteps)):
    obs = observations[i]
    filter.predict()
    means += (filter.x,)
    covs += (filter.P,)
    filter.update(obs)
means = np.array(means)
covs = np.array(covs)

end = time.time()
print("Elapsed time: ", end - start)
exit()


fig, ax = plt.subplots((1), figsize=(10, 4))
ax.set_title("Kalman filter")


for i, name, color in zip(range(3), ["pos", "vel", "acc"], ["red", "green", "blue"]):
    first = means[:, i]
    std = np.sqrt(covs[:, i, i])

    ax.plot(timesteps, first, label=f"predicted {name}", color=color)
    ax.fill_between(
        timesteps, first - std, first + std, label="stds", alpha=0.5, color=color
    )


# n_samples = 10
# samples = []
# start = np.array([0.0, 0.0, 0.0])
# for i in range(n_samples):
#     state = start
#     sample = [[0.0]]
#     for i in range(len(timesteps)-1):
#         state = A @ state + np.random.multivariate_normal([0, 0, 0], Q)
#         sample += H @ state + np.random.multivariate_normal([0], R),
#     samples += sample,

# # show samples
# for sample in samples:
#     ax.plot(timesteps, sample, label="sample", color="blue", alpha=0.1)


ax.plot(timesteps, observations, label="truth", color="black")

fig.legend()
plt.show()
