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
nb_cortical_units = 3  # Number of cortical neurons in layer 5
nb_l23_units = 3  # Number of cortical neurons in layer 2/3
time_step = 1e-3
nb_steps = 100
"""
spk2_rec data = tensor within tensors, the inner tensor is comprised of tensors again, with number of rows = number of 
integration steps set, and number of rows = number of output nodes (neurons). 
"""
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
y = np.sin(0.05 * x) * 35 + 15
# Format this data into a tensor
data = np.array(list(zip(x, y)))
tensor_sine_data = torch.from_numpy(data)

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

# Weights from layer 5 to 2/3
w5_23 = torch.empty((nb_thalamic_units, nb_cortical_units), dtype=dtype, device=device, requires_grad=True)
torch.nn.init.normal_(w5_23, mean=0.0, std=weight_scale / np.sqrt(nb_thalamic_units))


# Heaviside function
def spike_fn(x):  # Input is a tensor
    thr = 0
    spk = 1.0
    out = torch.zeros_like(x)  # Sets variable out to a tensor of zeros with same dimension as input tensor
    out[x > thr] = spk  # For each entry out, if the corresponding entry in x is greater than 0, set that entry in x = 1
    return out  # Return the tensor x, filled with zeros and 1s in the positions of spikes


def plot_voltage_traces(mem, spk=None, dim=(3, 5), spike_height=5):
    """
    :param mem: Tensor of membrane potentials (shape?)
    :param spk: No spikes ?
    :param dim: Dimension of the plots
    :param spike_height: Peak amplitude of each spike
    :return: Membrane potentials and spikes of output neurons (right?)
    """
    gs = GridSpec(*dim)  # Grid layout to place subplots within a figure
    if spk is not None:
        dat = mem * 1.0  # tensor - why multiply by 1.0?
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
    """
    Defines the SNN, iterates through the time steps and updates the dynamics at each step
    :param inputs: Tensor of input data
    :return: tensors of membrane potentials and spike records
    """
    # h1 = torch.einsum("abc,cd->abd", (inputs, w1))  # Computing hidden layer, could probably use torch.matmul?
    h1 = torch.matmul(inputs, w1)  # Compute hidden layer (THA units)
    syn = torch.zeros((batch_size, nb_thalamic_units), device=device, dtype=dtype)  # Initializing synaptic current
    mem = torch.zeros((batch_size, nb_thalamic_units), device=device, dtype=dtype)  # Initializing membrane potential
    mem_rec = []  # List to record membrane potentials at each time step for each neuron in hidden layer
    spk_rec = []  # List to record spikes at each time step for each neuron in hidden layer
    # Check if this is right^?^^?

    # Compute hidden layer activity
    for t in range(nb_steps):
        mthr = mem - 1.0  # Sets 'mthr' tensor = (initially) zeros mem tensor - 1.0 = tensor of all -1s initially
        # What is the purpose of this mthr tensor?
        out = spike_fn(mthr)  # Returns a tensor that spike function
        rst = out.detach()  # We do not want to backprop through the reset

        new_syn = alpha * syn + h1[:, t]  # I(t+1) = alpha*I(t) + input =
        new_mem = (beta * mem + syn) * (1.0 - rst)  # U(t+1) = beta*U(t) + I(t) * (1 - reset)
        # whenever there is an output spike (=1), rst = 1 => the whole thing gets reset to 0 (resting potential)

        mem_rec.append(mem)  # Add U(t) at each step for each neuron?
        spk_rec.append(out)  #

        mem = new_mem  # Updates time step, t = t+1, for next iteration (for U(t) here)
        syn = new_syn  # Same as above (but for I(t) here)

    mem_rec = torch.stack(mem_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)

    ############# Spiking output layer #############
    h2 = torch.matmul(spk_rec, w2)  # h2 = cortical layer 5
    syn2 = torch.zeros((batch_size, nb_cortical_units), device=device, dtype=dtype)
    mem2 = torch.zeros((batch_size, nb_cortical_units), device=device, dtype=dtype)

    mem2_rec = []
    spk2_rec = []

    for t in range(nb_steps):
        mthr2 = mem2 - 1.0
        out2 = spike_fn(mthr2)
        rst2 = out2.detach()

        new_syn2 = alpha * syn2 + h2[:, t]
        new_mem2 = (beta * mem2 + syn2) * (1.0 - rst2)

        mem2_rec.append(mem2)
        spk2_rec.append(out2)

        mem2 = new_mem2
        syn2 = new_syn2

    mem2_rec = torch.stack(mem2_rec, dim=1)
    spk2_rec = torch.stack(spk2_rec, dim=1)

    return mem2_rec, spk2_rec

    ############# Non-spiking output layer #############


'''    h2 = torch.einsum("abc,cd->abd", (spk_rec, w2))  # Computes read out layer = h2????
    # h2 = torch.matmul(spk_rec, w2)  # why are we putting spk_rec as input for output layer?
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
'''

mem2_rec, spk2_rec = run_snn(x_data)

'''fig = plt.figure(dpi=100)
plot_voltage_traces(mem2_rec, spk2_rec)'''


spikes = []
for i in spk2_rec:
    for j in i:
        for k in j:
            if k == 1:
                spikes.append(k)


# Pretty sure that iterates through the entire spk2_rec tensor and pulls out total number of spikes over the entire simulation


# Introducing surrogate gradients for training and setting this function = previous spike_fn(x) function
class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking non-linearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


# here we overwrite our naive spike function by the "SurrGradSpike" non-linearity which implements a surrogate gradient
spike_fn = SurrGradSpike.apply

params = [w2]  # Does this account for the recurrent connectivity between hidden and output layers? Since
# the recurrence is only written into the equations not the weights
