import torch
from torch.nn import *
torch.set_default_dtype(torch.float64)
torch.manual_seed(123)

N_Width = 60
delta = 1.0e-2
LR = 1.0e-2
Num_Epochs = 30000
activ = Sigmoid()

loss = MSELoss()

par_l = torch.tensor(1.0)
par_g = torch.tensor(9.81)
M = 501
numPeriod = 5
a = torch.tensor(2.0).reshape([1])
b = torch.tensor(-0.1).reshape([1])
t0 = 2 * torch.pi * torch.sqrt(par_l/par_g)
ts = torch.linspace(0.0, t0 * numPeriod, steps = M).reshape((M, 1))

model = Sequential(
    Linear(1, N_Width), activ,
    Linear(N_Width, N_Width), activ,
    Linear(N_Width, N_Width), activ,
    Linear(N_Width, N_Width), activ,
    Linear(N_Width, 1)
)
optimizer = torch.optim.Adam(model.parameters(), lr = LR)

for STEP in range(Num_Epochs):
    f = (model(ts + delta) + model(ts - delta) - 2 * model(ts)) / delta**2
    lossVal = loss(f, - par_g/par_l * model(ts))+\
        100.0 * loss(model(torch.zeros([1])), a) +\
        50.0 * loss((model(torch.tensor([delta])) - model(torch.zeros([1]))) / delta, b)
    if STEP % 100 == 0:
         print(STEP, lossVal.item())
         if STEP % 1000 == 0:
             torch.save([model, a, b], "model.pt")
    optimizer.zero_grad()
    lossVal.backward()
    optimizer.step()

torch.save([model, a, b], "model.pt")