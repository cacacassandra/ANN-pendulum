import torch
from torch.nn import *
torch.set_default_dtype(torch.float64)

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