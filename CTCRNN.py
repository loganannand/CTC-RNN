import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

nb_basalganglia = 10  # Number of BG neurons
nb_thalamic_units = 10  # Number of thalamic neurons (try to group them in 5s = 6 groups)
nb_cortical_units = 3  # Number of cortical neurons (representing layer 5)

time_step = 1e-3
nb_steps = 200

dtype = torch.float
device = torch.device("cpu")

# Synthetic data
batch_size = 256

freq = 5  # Hz
prob = freq * time_step
mask = torch.rand((batch_size, nb_steps, nb_basalganglia), device=device, dtype=dtype)
x_data = torch.zeros((batch_size, nb_steps, nb_basalganglia), device=device, dtype=dtype, requires_grad=False)
x_data[mask < prob] = 1.0

# Visualise synthetic data
'''data_id = 0
plt.imshow(x_data[data_id].cpu().t(), cmap=plt.cm.gray_r, aspect="auto")
plt.xlabel("Time (ms)")
plt.ylabel("Unit")
sns.despine()
plt.show()'''

# Parameters from the RNN equations defined in the spytorch jupyter notebook
tau_mem = 10e-3  # Membrane time constant
tau_syn = 5e-3  # Synaptic decay time constant
alpha = float(np.exp(-time_step / tau_syn))
beta = float(np.exp(-time_step / tau_mem))

weight_scale = 7 * (1.0 - beta)

# Weights from BG to THA
w1 = torch.zeros((nb_basalganglia, nb_thalamic_units), dtype=dtype, device=device)
# torch.nn.init.normal(w1, mean=0.0, std=weight_scale/np.sqrt(nb_basalganglia))
w1[0][0] = 1.0  # S1
'''w1[1][1] = 1.0  # S2
w1[5][5] = 1.0  # S5'''

# Weights from THA to Cx
w2 = torch.empty((nb_thalamic_units, nb_cortical_units), dtype=dtype, device=device, requires_grad=True)
torch.nn.init.normal_(w2, mean=0.0, std=weight_scale / np.sqrt(nb_thalamic_units))

h1 = torch.einsum("abc,cd->abd", (x_data, w1))  # Multiply input spikes with weight matrix


# Heaviside function
def spike_fn(x):
    out = torch.zeros_like(x)
    out[x > 0] = 1.0
    return out


syn = torch.zeros((batch_size, nb_thalamic_units), dtype=dtype, device=device)
mem = torch.zeros((batch_size, nb_thalamic_units), dtype=dtype, device=device)

mem_rec = []  # Record membrane potentials
spk_rec = []  # Record spikes

for t in range(nb_steps):
    mthr = mem - 1.0
    out = spike_fn(mthr)
    rst = out.detach()

    new_syn = alpha * syn + h1[:, t]
    new_mem = (beta * mem + syn) * (1.0 - rst)

    mem_rec.append(mem)
    spk_rec.append(out)

    mem = new_mem
    syn = new_syn

mem_rec = torch.stack(mem_rec, dim=1)
spk_rec = torch.stack(spk_rec, dim=1)


def plot_voltage_traces(mem, spk=None, dim=(3, 5), spike_height=5):
    gs = GridSpec(*dim)
    if spk is not None:
        dat = 1.0 * mem
        dat[spk > 0.0] = spike_height
        dat = dat.detach().cpu().numpy()
    else:
        dat = mem.detach().cpu().numpy()
    for i in range(np.prod(dim)):
        if i == 0:
            a0 = ax = plt.subplot(gs[i])
        else:
            ax = plt.subplot(gs[i], sharey=a0)
        ax.plot(dat[i])
        ax.axis("off")
    plt.show()


fig = plt.figure(dpi=100)
plot_voltage_traces(mem_rec, spk_rec)
