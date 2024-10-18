import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from numpy.random import randn

# Pendulum parameters
g = 9.81      # Acceleration due to gravity (m/s^2)
l = 1.0       # Length of the pendulum (m)
m = 1.0       # Mass of the pendulum bob (kg)
dt = 0.01     # Time step (s)
t = np.arange(0, 10, dt)  # Time vector

# True state initialization
theta = 0.1    # Initial angle (rad)
theta_dot = 0.0  # Initial angular velocity (rad/s)
true_states = []

# Measurement initialization
measurements = []

# Process and measurement noise
process_noise_std = np.array([1e-5, 1e-5])
measurement_noise_std = 0.1

# Control input (random torque)
np.random.seed(42)
tau_values = np.random.normal(0, 0.1, size=len(t))  # Random torques

# Simulation of true states and measurements
for i in range(len(t)):
    # Random torque at time t
    tau = tau_values[i]
    
    # Compute acceleration
    theta_double_dot = (-g / l) * np.sin(theta) + tau / (m * l**2)
    
    # Update state using Euler method
    theta_dot += theta_double_dot * dt
    theta += theta_dot * dt
    
    # Save true state
    true_states.append([theta, theta_dot])
    
    # Simulate measurement with noise
    measurement = theta + np.random.normal(0, measurement_noise_std)
    measurements.append(measurement)

# Convert to numpy arrays
true_states = np.array(true_states)
measurements = np.array(measurements)

# Extended Kalman Filter Implementation
def state_transition_function(x, u):
    """Non-linear state transition function."""
    theta = x[0]
    theta_dot = x[1]
    tau = u[0]
    
    # Compute theta_double_dot
    theta_double_dot = (-g / l) * np.sin(theta) + tau / (m * l**2)
    
    # Predict next state
    theta_new = theta + theta_dot * dt
    theta_dot_new = theta_dot + theta_double_dot * dt
    
    return np.array([theta_new, theta_dot_new])

def jacobian_F(x, u):
    """Jacobian of the state transition function with respect to x."""
    theta = x[0]
    theta_dot = x[1]
    
    F = np.array([
        [1, dt],
        [(-g / l) * np.cos(theta) * dt, 1]
    ])
    return F

def measurement_function(x):
    """Measurement function h(x)."""
    theta = x[0]
    return np.array([theta])

def jacobian_H(x):
    """Jacobian of the measurement function with respect to x."""
    return np.array([[1, 0]])

# Initialize EKF
ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
ekf.x = np.array([0.0, 0.0])  # Initial state estimate
ekf.P *= 1.0                  # Initial covariance estimate
ekf.R = np.array([[measurement_noise_std**2]])  # Measurement noise covariance
ekf.Q = np.diag(process_noise_std**2)           # Process noise covariance

# Storage for estimates
estimated_states = []

# Run EKF
for i in range(len(t)):
    # Control input (torque)
    u = np.array([tau_values[i]])
    
    # Predict
    ekf.predict_update(z=measurements[i], 
                       u=u, 
                       fx=state_transition_function, 
                       hx=measurement_function, 
                       F_jacobian=jacobian_F, 
                       H_jacobian=jacobian_H)
    
    # Save estimates
    estimated_states.append(ekf.x.copy())

# Convert to numpy arrays
estimated_states = np.array(estimated_states)

# Plotting the results
plt.figure(figsize=(12, 8))

# Angle plot
plt.subplot(2, 1, 1)
plt.plot(t, true_states[:, 0], label='True Angle')
plt.plot(t, measurements, 'r.', markersize=2, label='Measurements')
plt.plot(t, estimated_states[:, 0], 'g-', label='EKF Estimate')
plt.title('Pendulum Angle')
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.legend()

# Angular velocity plot
plt.subplot(2, 1, 2)
plt.plot(t, true_states[:, 1], label='True Angular Velocity')
plt.plot(t, estimated_states[:, 1], 'g-', label='EKF Estimate')
plt.title('Pendulum Angular Velocity')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')
plt.legend()

plt.tight_layout()
plt.show()
