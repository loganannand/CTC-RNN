
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

dtype = torch.float
device = torch.device("cpu")

nb_bg_units = 5
nb_tha_units = 5
time_step = 1e-3
nb_steps = 100
tau_mem = 10e-3  # Membrane time constant
tau_syn = 5e-3  # Synaptic decay time constant
alpha = float(np.exp(-time_step / tau_syn))
beta = float(np.exp(-time_step / tau_mem))

# Synthetic data
'''freq = 5  # Hz
prob = freq * time_step
torch.manual_seed(1542)
mask = torch.rand((nb_steps, nb_bg_units), device=device, dtype=dtype)
x_data = torch.zeros((nb_steps, nb_bg_units), device=device, dtype=dtype, requires_grad=False)
x_data[mask < prob] = 1.0'''

x_data = torch.empty((nb_steps, nb_bg_units), dtype=dtype, device=device)
torch.nn.init.normal_(x_data, mean=0.0, std=0.025)

####### FFN #######
Jtbg = torch.zeros((nb_bg_units, nb_tha_units), dtype=dtype, device=device)
torch.nn.init.normal_(Jtbg, mean=0.0, std=0.025)
# Selection of motif mu:
# Jtbg[0][0] = 1.0
tha_layer = torch.matmul(x_data, Jtbg)

thr = 0
spk = 1.0

'''
def spike_fn(x):
    out = torch.zeros_like(x)
    out[x > thr] = spk
    return out


I = torch.zeros((batch_size, nb_tha_units), dtype=dtype, device=device)
U = torch.zeros((batch_size, nb_tha_units), dtype=dtype, device=device)

spk_rec = []
U_rec = []

for t in range(nb_steps):
    memthresh = U - spk
    out = spike_fn(memthresh)
    reset = out.detach()

    new_I = alpha * I + tha_layer[:, t]
    new_U = (beta * U + I) * (spk - reset)

    U_rec.append(U)
    spk_rec.append(out)

    U = new_U
    I = new_I

U_rec = torch.stack(U_rec, dim=1)
spk_rec = torch.stack(spk_rec, dim=1)


def plot_voltage_traces(U, spk=None, dim=(3, 5), spike_height=5):
    gs = GridSpec(*dim)  # Grid layout to place subplots within a figure
    if spk is not None:
        dat = U * 1.0  # tensor - why multiply by 1.0?
        dat[spk > 0.0] = spike_height
        dat = dat.detach().cpu().numpy()
    else:
        dat = U.detach().cpu().numpy()
    for i in range(np.prod(dim)):
        if i == 0:
            a0 = ax = plt.subplot(gs[i])
        else:
            ax = plt.subplot(gs[i], sharey=a0)
        ax.plot(dat[i])
        ax.axis("off")
    plt.show()


fig = plt.figure(dpi=100)
plot_voltage_traces(U_rec, spk_rec)

####### RNN #######
'''