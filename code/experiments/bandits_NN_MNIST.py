import numpy as np
from tqdm import tqdm

from function import TridiagonalQuadraticFunction, StochasticTridiagonalQuadraticFunction
from function import StochasticQuadraticFunction
from theoritical.stepsizes import MindFlayerStepSize, RennalaStepSize
from function import generate_random_vector, create_worst_case
from asynchronous.asynchronous_transport import RandomDelayedAsynchronousTransport, DelayedAsynchronousTransport
from asynchronous.algorithm import StochasticGradientNodeAlgorithm, AsynchronousSGD,\
        RennalaSGD, RennalaWithBandits, MiniBatchSGD, AsynchronousTimeFrameMiniBatchSGD, DynamicClippingRennala, ClippingRennala, ATA, RennalaWithFixedBatchSizes, Rennala_EG_RR
from signature import Signature
from utils import harmonia
import pickle
import os


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from function import NeuralNetworkFunction, NeuralNetworkFunction_Data_Loader


"""
Hyperparameters
"""
batch_size = 23
nn_batch_size = 64
num_nodes = 17 # 17 51 153 459

gamma = 0.0001
seed = 5


"""
Time Setup
"""
generator = np.random.default_rng(seed=seed)
beta = 29

# time_means = np.array([2*beta*np.log(i+2) for i in range(num_nodes)], dtype='float')
# time_setup = f'{beta}log(i+2)+Exp({beta}log(i+2))'

# time_means = np.array([2*beta*np.sqrt(i+1) for i in range(num_nodes)], dtype='float')
# time_setup = f'{beta}sqrt(i+1)+Exp({beta}sqrt(i+1))'

time_means = np.array([2*beta*(i+1) for i in range(num_nodes)], dtype='float')
time_setup = f'{beta}(i+1)+Exp({beta}(i+1))'

# time_means = np.array([2*beta*(i+1)**2 for i in range(num_nodes)], dtype='float')
# time_setup = f'{beta}(i+1)**2+Exp({beta}(i+1)**2)'

def exponential(index):
    i = time_means[index] / 2
    t = i + generator.exponential(scale=i)
    return t
noise_function = exponential
alpha = 2*time_means.max()
number_of_gradients = harmonia(time_means, batch_size)
assert number_of_gradients.sum() == batch_size
print('Optimal Allocation:', number_of_gradients)

# it_lim = 3 * 1e6
it_lim = 6 * 1e3

"""
NN Function Setup
"""
    
# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=nn_batch_size, shuffle=True)

neural_network_function = NeuralNetworkFunction_Data_Loader(train_loader=train_loader, number_of_classes=10, neural_network_name='two_layer_neural_net_relu')

"""
Transport Setup
"""
nodes = [Signature(StochasticGradientNodeAlgorithm, neural_network_function) for _ in range(num_nodes)]
transport = RandomDelayedAsynchronousTransport(nodes, np.zeros(num_nodes), noise_function)

"""
Run Algorithms
"""
dim = neural_network_function.dim()
point = np.zeros(dim)

algorithms = [
    # ("ATA", ATA(transport, point, gamma=gamma, batch_size=batch_size, negative_strategy='uniform', alpha=alpha, print_aloc=False, time_means=time_means)),
    # ("ATA: Empirical", ATA(transport, point, gamma=gamma, batch_size=batch_size, negative_strategy='uniform', alpha='empirical', print_aloc=False, time_means=time_means)),
    # ("FTA: Optimal", ATA(transport, point, gamma=gamma, allocation_type='fixed', number_of_gradients=number_of_gradients, batch_size=batch_size, time_means=time_means)),
    ("GTA", RennalaSGD(transport, point, gamma=gamma, batch_size=batch_size)),
    # ("UTA", ATA(transport, point, gamma=gamma, batch_size=batch_size, allocation_type='uniform', alpha=alpha,time_means=time_means)),
]

run_setup = f'NN_n={num_nodes}_t={time_setup}_B={batch_size}_dim={dim}_gamma={gamma}_seed={seed}'
# Create the runs directory if it doesn't exist
run_dir = os.path.join(os.path.dirname(__file__), 'runs')
os.makedirs(run_dir, exist_ok=True)
# Create a directory for the run setup using the file name
setup_dir = os.path.join(os.path.dirname(__file__), 'runs', run_setup)
os.makedirs(setup_dir, exist_ok=True)

for name, optimizer in algorithms:
    iteration = 0
    iteration_times = [0]
    expected_iteration_time = [0]
    total_worker_time = [0]
    allocations = []
    loss = [neural_network_function.value(point)]
    # iteration_grads = [np.linalg.norm(function.gradient(point))**2]
    with tqdm(total=it_lim, desc=name, unit='iter') as pbar:
        while iteration < it_lim:
            optimizer.step()
            loss.append(neural_network_function.value(optimizer.get_point()))
            iteration_times.append(optimizer.get_time())
            total_worker_time.append(optimizer.get_total_worker_time())
            if name != 'GTA':
                expected_iteration_time.append(expected_iteration_time[-1] + optimizer.get_mean_time())
            if 'ATA' in name and (iteration % (it_lim // 200) == 0):
                allocations.append(optimizer.get_allocation())
            # iteration_grads.append(np.linalg.norm(function.gradient(optimizer.get_point())) ** 2)
            # Update the progress bar
            pbar.update(1)
            iteration += 1
    
    file_path = os.path.join(os.path.dirname(__file__), 'runs', run_setup, f'{name}.pkl')
    with open(file_path, 'wb') as file:
        result = (iteration_times, loss, total_worker_time, allocations, optimizer.get_warm_start() if 'ATA' in name else None, expected_iteration_time)
        pickle.dump(result, file)
