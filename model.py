import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Parameters
time_step = 1e-3  # 1 ms = 0.001 s
nb_steps = 200
time = time_step * nb_steps  # 0.2 s
nb_l23_neurons = 10
nb_l5_neurons = 10

dtype = torch.float
device = torch.device("cpu")

tau_mem = 10e-3  # membrane time constant - MIGHT NEED TO CHANGE VALUE
tau_syn = 5e-3  # synaptic decay time constant - MIGHT NEED TO CHANGE VALUE

alpha = float(np.exp(-time_step / tau_syn))
beta = float(np.exp(-time_step / tau_mem))


def spike_fn(x):  # Input is a tensor
    thr = 0
    spk = 1.0
    out = torch.zeros_like(x)  # Sets variable out to a tensor of zeros with same dimension as input tensor
    out[x > thr] = spk  # For each entry out, if the corresponding entry in x is greater than 0, set that entry in x = 1
    return out  # Return the tensor x, filled with zeros and 1s in the positions of spikes


# Weight matrices
Ji_23 = torch.empty((nb_l23_neurons, nb_l23_neurons), dtype=dtype, device=device)  # input -> l23 weights
J23_5 = torch.empty((nb_l23_neurons, nb_l5_neurons), dtype=dtype, device=device)  # l23 -> l5 weights

def run_snn(inputs):
    # define layer 2/3 neurons
    layer_23 = torch.matmul(inputs, Ji_23)

    # initialize synapse and membrane values for layer 2/3, i.e., I(t=0) and U(t=0), respectively
    l23_syn = torch.zeros((nb_l23_neurons, nb_l23_neurons), device=device, dtype=dtype)
    l23_mem = torch.zeros((nb_l23_neurons, nb_l23_neurons), device=device, dtype=dtype)

    # record membrane potential and spikes in layer 2/3
    l23_mem_rec = []
    l23_spk_rec = []

    # Compute layer 2/3 activity
    for t in range(nb_steps):
        mthr = mem - 1.0  # Sets 'mthr' tensor = (initially) zeros mem tensor - 1.0 = tensor of all -1s initially
        # What is the purpose of this mthr tensor?
        out = spike_fn(mthr)  # Returns a tensor from the spike function
        rst = out.detach()  # We do not want to backprop through the reset

        new_l23_syn = alpha * l23_syn + layer_23[:, t]  # not sure if that indexing is correct???
        new_l23_mem = (beta * l23_mem + l23_syn) * (1.0 - rst)

        l23_mem_rec.append(l23_mem)
        l23_spk_rec.append(out)

        l23_mem = new_l23_mem
        l23_syn = new_l23_syn

    l23_mem_rec = torch.stack(l23_mem_rec, dim=1)
    l23_spk_rec = torch.stack(l23_spk_rec, dim=1)

    # Compute layer 5 activity
    l5 = torch.matmul(l23_spk_rec, J23_5)  # layer 5

    # initialize synapse and membrane values for layer 5, i.e., I(t=0) and U(t=0), respectively
    l5_syn = torch.zeros((nb_l23_neurons, nb_l5_neurons), device=device, dtype=dtype)
    l5_mem = torch.zeros((nb_l23_neurons, nb_l5_neurons), device=device, dtype=dtype)

    out_rec = [out]
    for t in range(nb_steps):
        new_l5_syn = alpha * l5_syn + l5[:, t] # not sure if that indexing is correct???


