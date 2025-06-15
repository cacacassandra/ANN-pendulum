import torch
from torch.nn import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

torch.set_default_dtype(torch.float64)


# Define the neural network architecture
class PendulumNN(Module):
    def __init__(self, n_width, n_layers=4):
        super(PendulumNN, self).__init__()
        layers = []
        layers.append(Linear(1, n_width))
        layers.append(Tanh())
        for _ in range(n_layers - 1):
            layers.append(Linear(n_width, n_width))
            layers.append(Tanh())
        layers.append(Linear(n_width, 1))
        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Pendulum parameters
g = 9.81  # acceleration due to gravity (m/s^2)
L = 1.0  # length of the pendulum (m)


# Define the differential equation for the pendulum
def pendulum_ode(model, x, delta):
    theta = model(x)
    theta_plus = model(x + delta)
    theta_minus = model(x - delta)

    # Approximation of the second derivative (theta'')
    theta_double_prime = (theta_plus + theta_minus - 2 * theta) / delta ** 2

    # The residual of the differential equation: theta'' + (g/L) * sin(theta) = 0
    f = theta_double_prime + (g / L) * torch.sin(theta)

    return f


# Weight initialization function
def init_weights(m):
    if isinstance(m, Linear):
        torch.nn.init.xavier_uniform_(m.weight)


# Hyperparameters
N_Width = 50
delta = 1.0e-2
LR = 1.0e-4  # Further reduced learning rate
Num_Epochs = 200000  # Further increased number of epochs
M = 200

# Initialize the neural network
model = PendulumNN(N_Width, n_layers=5)  # Increased the number of layers
model.apply(init_weights)
loss_function = MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                             weight_decay=1e-5)  # Added weight decay

# Initial conditions: theta(0) = theta_0, theta'(0) = theta_prime_0
theta_0 = torch.tensor([0.1])  # Small initial angle in radians
theta_prime_0 = torch.tensor([0.0])  # Initial angular velocity

# Training loop
for STEP in range(Num_Epochs):
    x = torch.linspace(0, 10, M).reshape(
        [M, 1])  # Uniformly sampled time points

    # Calculate the differential equation residual
    f = pendulum_ode(model, x, delta)
    target = torch.zeros_like(f)

    # Boundary condition losses
    theta_at_0 = model(torch.tensor([0.0]).reshape([1]))
    theta_prime_at_0 = (model(
        torch.tensor([delta]).reshape([1])) - theta_at_0) / delta

    # Total loss
    loss = loss_function(f, target) + \
           200 * loss_function(theta_at_0, theta_0) + \
           200 * loss_function(theta_prime_at_0,
                               theta_prime_0)  # Increased the weight of initial condition terms

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 1000 steps
    if STEP % 1000 == 0:
        print(f"Step {STEP}: Loss = {loss.item()}")

# Save the trained model
torch.save(model, "pendulum_model.pt")

# Load the model and generate the plot
model = torch.load("pendulum_model.pt")


# Numerical solution for comparison (simple example using small angle approximation)
def numerical_solution(t, theta_0, omega_0, g, L):
    g_tensor = torch.tensor(g)  # Convert g to a tensor
    L_tensor = torch.tensor(L)  # Convert L to a tensor

    omega = omega_0 * torch.cos(
        torch.sqrt(g_tensor / L_tensor) * t) + theta_0 * torch.sin(
        torch.sqrt(g_tensor / L_tensor) * t)
    return omega


N = 500
t = torch.linspace(0, 10, steps=N)
theta_nn = model(t.reshape([N, 1])).detach().reshape([N])
theta_num = numerical_solution(t, theta_0, theta_prime_0, g, L)

# Plot the results
with PdfPages("pendulum_output.pdf") as pdf:
    plt.figure()
    plt.plot(t, theta_num, color="red", label="Numerical Solution")
    plt.plot(t, theta_nn, color="blue", label="Neural Network Output")
    plt.title("Nonlinear Pendulum: Comparison of Solutions")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (radians)")
    plt.legend()
    pdf.savefig()
    plt.close()

print("Plot saved to pendulum_output.pdf")
