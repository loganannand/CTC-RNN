import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

nb_basalganglia = 10  # Number of BG neurons
nb_thalamic_units = 30  # Number of thalamic neurons (try to group them in 5s = 6 groups)
nb_cortical_units = 3 # Number of cortical neurons (representing layer 5)

time_step = 1e-3
nb_steps = 200

dtype = torch.float
device = torch.device("cpu")

# Parameters from the RNN equations defined in the spytorch jupyter notebook
tau_mem = 10e-3  # Membrane time constant
tau_syn = 5e-3  # Synaptic decay time constant
alpha = float(np.exp(-time_step/tau_syn))
beta = float(np.exp(-time_step/tau_mem))

weight_scale = 7*(1.0-beta)

# Weights from BG to THA
w1 = torch.empty((nb_basalganglia, nb_thalamic_units), dtype=dtype, device=device, requires_grad=True)  # Weights
torch.nn.init.normal(w1, mean=0.0, std=weight_scale/np.sqrt(nb_basalganglia))  # Initialize weights





