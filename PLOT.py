import torch
from torch.nn import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


torch.set_default_dtype(torch.float64)
model, a, b = torch.load("model.pt")
model.eval()

par_l = torch.tensor(1.0)
par_g = torch.tensor(9.81)
C1 = a
C2 = b * torch.sqrt(par_l / par_g)

M = 2000
num_periods = 5
t0 = 2 * torch.pi * torch.sqrt(par_l / par_g)
ts = torch.linspace(0.0, t0 * num_periods, steps=M).reshape((M, 1))
dt = (ts[1] - ts[0]).item()

# === 1. True Solution  ===
temp = torch.sqrt(par_g / par_l) * ts
ys = C1 * torch.cos(temp) + C2 * torch.sin(temp)
ys = ys.reshape([M])
ts_flat = ts.reshape([M])

# === 2. ANN Prediction ===
with torch.no_grad():
    y_pred = model(ts).reshape([M])

# === 3. Euler Method ===
theta_euler = [a.item()]
omega_euler = [b.item()]
for i in range(1, M):
    prev_theta = theta_euler[-1]
    prev_omega = omega_euler[-1]
    new_theta = prev_theta + dt * prev_omega
    new_omega = prev_omega - dt * (par_g / par_l).item() * torch.sin(torch.tensor(prev_theta)).item()
    theta_euler.append(new_theta)
    omega_euler.append(new_omega)
theta_euler = torch.tensor(theta_euler)

# === 4. Runge-Kutta 4th Order (RK4) Method ===
theta_rk4 = [a.item()]
omega_rk4 = [b.item()]
for i in range(1, M):
    th = theta_rk4[-1]
    om = omega_rk4[-1]

    k1_th = om
    k1_om = -(par_g / par_l).item() * torch.sin(torch.tensor(th)).item()

    k2_th = om + 0.5 * dt * k1_om
    k2_om = -(par_g / par_l).item() * torch.sin(torch.tensor(th + 0.5 * dt * k1_th)).item()

    k3_th = om + 0.5 * dt * k2_om
    k3_om = -(par_g / par_l).item() * torch.sin(torch.tensor(th + 0.5 * dt * k2_th)).item()

    k4_th = om + dt * k3_om
    k4_om = -(par_g / par_l).item() * torch.sin(torch.tensor(th + dt * k3_th)).item()

    new_th = th + (dt / 6) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
    new_om = om + (dt / 6) * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)

    theta_rk4.append(new_th)
    omega_rk4.append(new_om)
theta_rk4 = torch.tensor(theta_rk4)

# === 5. Plotting ===
fig = PdfPages("output_all_methods.pdf")
plt.figure(figsize=(12, 6))
plt.plot(ts_flat, ys, label="True Solution", color="red", linestyle='--', linewidth=1.8)
plt.plot(ts_flat, y_pred, label="ANN", color="blue", linewidth=1.5)
plt.plot(ts_flat, theta_euler, label="Euler Method", color="purple", linewidth=1)
plt.plot(ts_flat, theta_rk4, label="RK4 Method", color="green", linewidth=1)
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Angle (radians)", fontsize=14)
plt.title("Pendulum: True vs ANN vs Euler vs RK4", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
fig.savefig()
fig.close()

print("Comparison plot saved to output_all_methods.pdf")
