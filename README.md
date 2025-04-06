# DEEP_LEARNING_LAB2_196
import math
import numpy as np
import matplotlib.pyplot as plt

# Implement Tanh function manually: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
def tanh_manual(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

# Input values
inputs = [-1.5, 0.0, 1.0, 2.0]

# Compute Tanh values for the given inputs
tanh_values = [tanh_manual(x) for x in inputs]
print("Manual Tanh values for inputs [-1.5, 0.0, 1.0, 2.0]:", tanh_values)

# Compare to expected output (approximately): [-0.905, 0.0, 0.762, 0.964]

# Sigmoid function for comparison
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Generate values for smooth plotting
x = np.linspace(-5, 5, 100)

# Apply Tanh and Sigmoid to the range of x
y_tanh = np.array([tanh_manual(xi) for xi in x])
y_sigmoid = 1 / (1 + np.exp(-x))

# Plot both Tanh and Sigmoid
plt.plot(x, y_tanh, label="Tanh", color='b')
plt.plot(x, y_sigmoid, label="Sigmoid", color='r', linestyle='--')
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.title("Tanh vs Sigmoid Activation Functions")
plt.legend()
plt.show()
