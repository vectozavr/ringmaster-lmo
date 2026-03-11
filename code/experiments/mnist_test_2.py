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

# ---------------------------
# Experiment setup
# ---------------------------
num_nodes = 100
time_lim = 40000
seed = 26
np.random.seed(seed)

# Dataset (deterministic)
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

# ---------------------------
# Delays: build once, deterministic
# ---------------------------
base_delays = np.array([1 + i for i in range(num_nodes)], dtype=int)
rng_delays = np.random.default_rng(seed=seed)
delays = rng_delays.permutation(base_delays)

# ---------------------------
# Noise factory: per-node deterministic streams
# Recreating this factory yields the same noise, independent of other runs.
# ---------------------------
def make_noise_function(num_nodes, seed):
    gens = [np.random.default_rng(seed + i) for i in range(num_nodes)]
    def halfnormal(index):
        return abs(gens[index].normal(loc=0.0, scale=np.sqrt(index + 1)))
    return halfnormal

# ---------------------------
# Transport factory: fresh nodes + fresh noise per run (isolated)
# ---------------------------
def make_transport(features, labels, client_X, client_y, delays, seed):
    nodes = [Signature(StochasticGradientNodeAlgorithm,
                       SimpleNeuralNetFunction(X, y, seed=seed))
             for (X, y) in zip(client_X, client_y)]
    noise_fn = make_noise_function(num_nodes=len(nodes), seed=seed)
    return RandomDelayedAsynchronousTransport(nodes, delays.copy(), noise_fn)

# ---------------------------
# Single method runner (isolated)
# ---------------------------
def run_method(OptClass, label, gamma, color, marker):
    # Fresh function so weights & RNG state are identical across methods
    function = SimpleNeuralNetFunction(features, labels, seed=seed)
    init_point = np.array(function.get_current_point(), copy=True)

    # Fresh transport (nodes + noise) for this method
    transport = make_transport(features, labels, client_X, client_y, delays, seed)

    optimizer = OptClass(transport, init_point, gamma=gamma)
    print(optimizer.__class__.__name__)

    iteration_grads = [np.linalg.norm(function.gradient(init_point)) ** 2]
    iteration_times = [0]

    while iteration_times[-1] < time_lim:
        optimizer.step()
        iteration_times.append(optimizer.get_time())
        iteration_grads.append(np.linalg.norm(function.gradient(optimizer.get_point())) ** 2)

    plt.semilogy(
        iteration_times, iteration_grads,
        label=f"{label}: $\\gamma={gamma}$",
        linestyle='solid', marker=marker, markersize=16,
        markevery=max(1, len(iteration_times) // 10), color=color
    )

# ---------------------------
# Run methods (apples-to-apples)
# ---------------------------
run_method(RingleaderASGD, "Ringleader ASGD", 0.05, colors[0], markers[0])
run_method(MaleniaSGD,     "Malenia SGD",     0.7,  colors[1], markers[1])
run_method(IAASGD,         r"$\mathrm{IA^2SGD}$", 0.01, colors[2], markers[2])

plt.legend(loc='upper right', prop={'size': 16})
plt.xlim(0, time_lim)
plt.ylim(1e-6, 1e2)
plt.xlabel("Runtime (seconds)")
plt.ylabel(r"$f(x^t)-f^{\inf}$")
plt.show()
# plt.savefig('plotik.pdf')
