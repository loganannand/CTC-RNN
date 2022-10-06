"""
Spiking RNN with C-T-C loops
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

# Parameters
nb_basalganglia = 10  # Number of BG neurons (input layer)
nb_thalamic_units = 10  # Number of thalamic neurons (try to group them in 5s = 6 groups) (hidden layer)
nb_cortical_units = 10  # Number of cortical neurons (representing layer 5) (output layer)
time_step = 1e-3
nb_steps = 100
dtype = torch.float
device = torch.device("cpu")
tau_mem = 10e-3  # Membrane time constant
tau_syn = 5e-3  # Synaptic decay time constant
alpha = float(np.exp(-time_step / tau_syn))
beta = float(np.exp(-time_step / tau_mem))

# Synthetic data
batch_size = 256
freq = 5  # Hz
prob = freq * time_step
torch.manual_seed(1729)
mask = torch.rand((batch_size, nb_steps, nb_basalganglia), device=device, dtype=dtype)
x_data = torch.zeros((batch_size, nb_steps, nb_basalganglia), device=device, dtype=dtype, requires_grad=False)
x_data[mask < prob] = 1.0

# Sine wave data for regression training
xlim = nb_steps
x = np.arange(0, xlim, time_step)
y = np.sin(0.05*x)
# Format this data into a tensor
data = np.array(list(zip(x, y)))
tensor_data = torch.from_numpy(data)

# Visualise synthetic data
'''data_id = 0
plt.imshow(x_data[data_id].cpu().t(), cmap=plt.cm.gray_r, aspect="auto")
plt.xlabel("Time (ms)")
plt.ylabel("Unit")
sns.despine()
plt.show()'''

weight_scale = 7 * (1.0 - beta)
# Weights from BG to THA
w1 = torch.zeros((nb_basalganglia, nb_thalamic_units), dtype=dtype, device=device)
torch.nn.init.normal_(w1, mean=0.0, std=weight_scale / np.sqrt(nb_basalganglia))
'''w1[0][0] = 1.0  # S1
w1[1][1] = 1.0  # S2
w1[5][5] = 1.0  # S5'''
# Weights from THA to Cx
w2 = torch.empty((nb_thalamic_units, nb_cortical_units), dtype=dtype, device=device, requires_grad=True)
torch.nn.init.normal_(w2, mean=0.0, std=weight_scale / np.sqrt(nb_thalamic_units))


# Heaviside function
def spike_fn(x):  # Input is a tensor
    thr = 0
    spk = 1.0
    out = torch.zeros_like(x)  # Sets variable out to a tensor of zeros with same dimension as input tensor
    out[x > thr] = spk  # For each entry out, if the corresponding entry in x is greater than 0, set that entry in x = 1
    return out  # Return the tensor x, filled with zeros and 1s in the positions of spikes

def plot_voltage_traces(mem, spk=None, dim=(3, 5), spike_height=5):
    gs = GridSpec(*dim)  # Grid layout to place subplots within a figure
    if spk is not None:
        dat = mem * 1.0
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


def run_snn(inputs):
    # h1 = torch.einsum("abc,cd->abd", (inputs, w1))  # Computing hidden layer, could probably use torch.matmul?
    h1 = torch.matmul(inputs, w1)  # Compute hidden layer
    syn = torch.zeros((batch_size, nb_thalamic_units), device=device, dtype=dtype)  # Initializing
    mem = torch.zeros((batch_size, nb_thalamic_units), device=device, dtype=dtype)

    mem_rec = []
    spk_rec = []

    # Compute hidden layer activity
    for t in range(nb_steps):
        mthr = mem - 1.0  # Sets 'mthr' tensor = (initially) zeros mem tensor - 1.0 = tensor of all -1s initially
        out = spike_fn(mthr)  # Runs that tensor through spike function
        rst = out.detach()  # We do not want to backprop through the reset

        new_syn = alpha * syn + h1[:, t]  # I(t+1) = alpha*I(t) + (?)
        new_mem = (beta * mem + syn) * (1.0 - rst)  # U(t+1) = beta*U(t) + I(t) * (1 - reset)
        # whenever there is an output spike (=1), rst = 1 => the whole thing gets reset to 0 (resting potential)

        mem_rec.append(mem)
        spk_rec.append(out)

        mem = new_mem
        syn = new_syn

    mem_rec = torch.stack(mem_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)
    # Gives activity of the hidden layer as a result of the input layer, i.e., THA activity caused by BG inputs

    # Output layer - Represents L5 for now
    # h2 = torch.einsum("abc,cd->abd", (spk_rec, w2))  # Computes read out layer = h2????
    h2 = torch.matmul(spk_rec, w2)
    flt = torch.zeros((batch_size, nb_cortical_units), device=device, dtype=dtype)  # Like new synaptic currents right?
    out = torch.zeros((batch_size, nb_cortical_units), device=device, dtype=dtype)
    out_rec = [out]

    for t in range(nb_steps):
        new_flt = alpha * flt + h2[:, t]  # Add recurrent connection here
        new_out = beta * out + flt  # Why is this eq different?

        flt = new_flt
        out = new_out

        out_rec.append(out)

    out_rec = torch.stack(out_rec, dim=1)
    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs



out_rec, other_recs = run_snn(x_data)

fig = plt.figure(dpi=100)
plot_voltage_traces(out_rec)
