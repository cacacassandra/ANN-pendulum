import torch
from torch.nn import *
torch.set_default_dtype(torch.float64)

# 2nd order ODE: y'' - 3y' - 4y = 0, where y(0) = 4, y'(0) = 6
N_Width = 50   # the number of neurons
delta = 1.0e-2
LR = 2.0e-2  #learning rate
Num_Epochs = 40000
M = 200
#activ = ReLU
activ = Sigmoid

model = Sequential(
 Linear(1, N_Width), activ(),
 Linear(N_Width, N_Width), activ(),
 Linear(N_Width, N_Width), activ(),
 Linear(N_Width, 1)
)

loss = MSELoss()   # mean squared error loss
T0 = torch.tensor(0.0).reshape([1])
T1 = torch.tensor(4.0).reshape([1])
T2 = torch.tensor(6.0).reshape([1])
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
tensor0 = torch.zeros([M, 1])

for STEP in range(Num_Epochs):
    x = torch.rand(M).reshape([M,1])
    #x = torch.linspace(0, 1, M).reshape([M,1])

# Updated loss calculation
    f = (model(x + delta) + model(x - delta) - 2 * model(x)) / delta**2 -\
        3 * (model(x + delta) - model(x)) / delta - 4 * model(x)
    lossVal = loss(f, tensor0)+\
              10 * loss(model(T0), T1) +\
              10 * loss((model(T0+delta) - model(T0)) / delta, T2)

    if STEP % 100 == 0:
         print(STEP, lossVal.item())
    optimizer.zero_grad() # clear accumulated gradients
    lossVal.backward()
    optimizer.step() # update model parameters

torch.save(model, "model.pt")


model = torch.load("model.pt")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

N = 500
x = torch.linspace(0, 1, steps = N)
y = model(x.reshape([N, 1])).detach().reshape([N])
y2 = 2 * (torch.exp(-x) + torch.exp(4 * x))

fig = PdfPages("output1.pdf")
plt.plot(x, y2, color = "red")
plt.plot(x, y, color = "blue")
plt.suptitle("red: true solution; blue: ANN output", y=0.98, fontsize=12)
fig.savefig()
fig.close()
