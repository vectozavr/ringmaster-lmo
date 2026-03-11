import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from function import create_worst_case
from asynchronous.asynchronous_transport import RandomDelayedAsynchronousTransport, DelayedAsynchronousTransport
from asynchronous.algorithm import StochasticGradientNodeAlgorithm, AsynchronousSGD, RingmasterASGD,\
        RennalaSGD, RingleaderASGD, MaleniaSGD, IAASGD
from signature import Signature
from utils_NN import prepare_dataset, StochasticNeuralNetworkFunction, NeuralNetworkFunction
from model import SimpleNeuralNetFunction
from prep_data import dirichlet_partition_equal_size, reduce_dataset_to_multiple

sns.set(style="whitegrid", context="talk", font_scale=1.2, palette=sns.color_palette("bright"), color_codes=False)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['figure.figsize'] = (10, 7)
matplotlib.rcParams['text.usetex'] = True

colors = [
    "#C00000",
    "#B8860B",
    "#006666",
]

markers = [
    '*',
    '^',
    'H',
    ]

"""
Function Setup: this is the quadratic function used in the paper
"""
num_nodes = 100
time_lim = 40000
seed = 26
np.random.seed(seed)

# features, labels, number_of_classes = prepare_dataset("./", 3000)
features, labels, number_of_classes = prepare_dataset("./", )
features, labels, kept_idx, dropped_idx = reduce_dataset_to_multiple(
    features, labels, number_of_classes, num_nodes, seed=seed
)
client_X, client_y, client_idxs = dirichlet_partition_equal_size(
    features, labels, number_of_classes,
    num_nodes=num_nodes, alpha=0.1, seed=seed
)
for i, y in enumerate(client_y):
    counts = np.bincount(y, minlength=number_of_classes)
    print(f"Client {i:2d}: {counts}")

function = SimpleNeuralNetFunction(features, labels, seed=seed)
nodes = [Signature(StochasticGradientNodeAlgorithm, SimpleNeuralNetFunction(X, y, seed=seed))
         for X, y in zip(client_X, client_y)]

"""
Time Setup: setup randomness for each node
"""
delays_1 = np.arange(2, num_nodes//2 + 2)
delays_2 = np.array([i**3 for i in range(num_nodes//2, num_nodes)])
delays = np.concatenate((delays_1, delays_2))
assert len(delays) == num_nodes
# delays = np.array([1 + i*4 for i in range(num_nodes)])
# delays = np.array([1 + np.sqrt(i) for i in range(num_nodes)])
# delays = np.array([1 + i**2 for i in range(num_nodes)])
delays = np.array([1 + i for i in range(num_nodes)])
# delays = np.array(2 + 20 * np.arange(num_nodes))
rng = np.random.default_rng(seed=seed)
delays = rng.permutation(delays)

generator = np.random.default_rng(seed=seed)
# sigma_normal = 10
def halfnormal(index):
    return np.abs(generator.normal(loc=0,scale=np.sqrt(index+1)))
    
noise_function = halfnormal

"""
Transport Setup: building random time for optimizers
"""
transport = RandomDelayedAsynchronousTransport(nodes, delays, noise_function)
init_point = np.array(function.get_current_point(), copy=True)

"""
Optimizer setup
"""

gamma=0.05
optimizer = RingleaderASGD(transport, init_point, gamma=gamma)
print(optimizer.__class__.__name__)

iteration_grads = [np.linalg.norm(function.gradient(init_point)) **2]
iteration_times = [0]


while iteration_times[-1] < time_lim:
    optimizer.step()
    iteration_times.append(optimizer.get_time())
    iteration_grads.append(np.linalg.norm(
        function.gradient(optimizer.get_point())) ** 2)

plt.semilogy(iteration_times, iteration_grads, label=rf"Ringleader ASGD: $\gamma={gamma}$", linestyle='solid', marker=markers[0], markersize=18, markevery=max(1, len(iteration_times) // 10), color=colors[1])

gamma=0.7
optimizer = MaleniaSGD(transport, init_point, gamma=gamma)
print(optimizer.__class__.__name__)

iteration_grads = [np.linalg.norm(function.gradient(init_point)) **2]
iteration_times = [0]

while iteration_times[-1] < time_lim:
    optimizer.step()
    iteration_times.append(optimizer.get_time())
    iteration_grads.append(np.linalg.norm(
        function.gradient(optimizer.get_point())) ** 2)

plt.semilogy(iteration_times, iteration_grads, label=rf"Malenia SGD: $\gamma={gamma}$", linestyle='solid', marker=markers[2], markersize=16, markevery=max(1, len(iteration_times) // 10), color=colors[2],)

gamma=0.01
optimizer = IAASGD(transport, init_point, gamma=gamma)
print(optimizer.__class__.__name__)

iteration_grads = [np.linalg.norm(function.gradient(init_point)) **2]
iteration_times = [0]

while iteration_times[-1] < time_lim:
    optimizer.step()
    iteration_times.append(optimizer.get_time())
    iteration_grads.append(np.linalg.norm(
        function.gradient(optimizer.get_point())) ** 2)

plt.semilogy(iteration_times, iteration_grads, label=rf"$\mathrm{{IA^2SGD}}$: $\gamma={gamma}$", linestyle='solid', marker=markers[2], markersize=16, markevery=max(1, len(iteration_times) // 10), color=colors[0],)

plt.legend(loc='upper right', prop={'size': 16})
plt.xlim(0, time_lim)
plt.ylim(1e-6, 1e2)
plt.xlabel("Runtime (seconds)")
plt.ylabel(r"$f(x^t)-f^{\inf}$")
plt.show()
# plt.savefig('plotik.pdf')