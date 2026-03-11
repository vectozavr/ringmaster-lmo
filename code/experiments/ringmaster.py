import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from function import StochasticTridiagonalQuadraticFunction
from function import create_worst_case
from asynchronous.asynchronous_transport import RandomDelayedAsynchronousTransport
from asynchronous.algorithm import StochasticGradientNodeAlgorithm, AsynchronousSGD, RingmasterASGD, RennalaSGD
from signature import Signature

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

def greed_search(gammas, max_delays, time_lim, optimizer_name):
    best_performance = np.inf
    for gamma in gammas:
        for max_delay in max_delays:
            if optimizer_name == "RingmasterASGD":
                optimizer = RingmasterASGD(transport, point, max_delay=max_delay, gamma=gamma)
            elif optimizer_name == "RennalaSGD":
                optimizer = RennalaSGD(transport, point, gamma=gamma, batch_size=max_delay)
            elif optimizer_name == "ASGD":
                optimizer = AsynchronousSGD(transport, point, gamma=gamma, delay_adaptive=True)
            
            x_0 = f(point) - f(analytical_solution)
            iteration_grads = [np.linalg.norm(function.gradient(point)) **2]
            iteration_points = [x_0]
            iteration_times = [0]

            while iteration_times[-1] < time_lim:
                optimizer.step()
                iteration_points.append(f(optimizer.get_point()) - f(analytical_solution))
                iteration_times.append(optimizer.get_time())
                iteration_grads.append(np.linalg.norm(
                    function.gradient(optimizer.get_point())) ** 2)

            performance = iteration_points[-1]
            print('performance:', performance,'gamma:', gamma,'max_delay:', max_delay)
            if performance < best_performance:
                best_performance = performance
                best_params = (gamma, max_delay)

    return best_params
        

"""
Function Setup: this is the quadratic function used in the paper
"""
dim = 1729
num_nodes = 6174

seed = 26  # seed correspondence to randomness in stochastic gradients
noise = 0.01
sigma2 = noise

time_lim = 2*1e3

# Function setup for known sigma
main_diag, side_diag, b = create_worst_case(dim, 1)
stochastic_func = StochasticTridiagonalQuadraticFunction(main_diag, side_diag, b,
                                                         seed, noise, "add")
function = stochastic_func._tridiagonal_quadratic

analytical_solution = np.linalg.solve(function._A.toarray(), function._b)
f = lambda x: 1/2 * x.T @ function._A @ x - function._b.T @ x
fx_star = f(analytical_solution)

print(fx_star)

"""
Time Setup: setup randomness for each node
"""
delays_1 = np.arange(2, num_nodes//2 + 2)
delays_2 = np.array([i**3 for i in range(num_nodes//2, num_nodes)])
delays = np.concatenate((delays_1, delays_2))
assert len(delays) == num_nodes
# delays = np.array([1 + i*4 for i in range(num_nodes)])
delays = np.array([1 + np.sqrt(i) for i in range(num_nodes)])
# delays = np.array([1 + i for i in range(num_nodes)])
# delays = np.array(2 + 20 * np.arange(num_nodes))
# np.random.shuffle(delays)

generator = np.random.default_rng(seed=5)
# sigma_normal = 10
def halfnormal(index):
    return np.abs(generator.normal(loc=0,scale=np.sqrt(index+1)))
    
noise_function = halfnormal

"""
Transport Setup: building random time for optimizers
"""
nodes = [Signature(StochasticGradientNodeAlgorithm, stochastic_func) for _ in range(num_nodes)]
transport = RandomDelayedAsynchronousTransport(nodes, delays, noise_function)


"""
Optimizer setup
"""
# initial point
point = np.zeros(dim)
point[0] = np.sqrt(dim)



# gammas = [5**i for i in range(-5, 5)]
# max_delays = []
# a = num_nodes
# while a >= 1:
#     max_delays.append(a)
#     a //= 4
# gamma, max_delay = greed_search(gammas=gammas, max_delays=max_delays, time_lim=time_lim, optimizer_name="RingmasterASGD")
# print(gamma, max_delay)

optimizer = RingmasterASGD(transport, point, max_delay=6, gamma=0.25)
print(optimizer.__class__.__name__)

x_0 = f(point) - f(analytical_solution)
iteration_grads = [np.linalg.norm(function.gradient(point)) **2]
iteration_points = [x_0]
iteration_times = [0]


while iteration_times[-1] < time_lim:
    optimizer.step()
    iteration_points.append(f(optimizer.get_point()) - f(analytical_solution))
    iteration_times.append(optimizer.get_time())
    iteration_grads.append(np.linalg.norm(
        function.gradient(optimizer.get_point())) ** 2)


plt.semilogy(iteration_times, iteration_grads, label=r"Ringmaster ASGD: $\gamma=0.2$, $R=6$", linestyle='solid', marker=markers[0], markersize=18, markevery=max(1, len(iteration_times) // 10), color=colors[0])


# gammas = [5**i for i in range(-5, 5)]
# batch_sizes = []
# a = num_nodes
# while a >= 1:
#     batch_sizes.append(a)
#     a //= 4
# gamma, batch_size = greed_search(gammas=gammas, max_delays=batch_sizes, time_lim=time_lim, optimizer_name="RennalaSGD")
# print(gamma, batch_size)
optimizer = RennalaSGD(transport, point, gamma=1, batch_size=6) # tuned parameters
print(optimizer.__class__.__name__)

x_0 = f(point) - f(analytical_solution)
iteration_grads = [np.linalg.norm(function.gradient(point)) **2]
iteration_points = [x_0]
iteration_times = [0]


while iteration_times[-1] < time_lim:
    optimizer.step()
    iteration_points.append(f(optimizer.get_point()) - f(analytical_solution))
    iteration_times.append(optimizer.get_time())
    iteration_grads.append(np.linalg.norm(
        function.gradient(optimizer.get_point())) ** 2)


plt.semilogy(iteration_times, iteration_grads, label=r"Rennala SGD: $\gamma=1$, $B=6$", linestyle='solid', marker=markers[2], markersize=16, markevery=max(1, len(iteration_times) // 10), color=colors[2],)


# gammas = [5**i for i in range(-5, 5)]
# gamma, max_delay = greed_search(gammas=gammas, max_delays=['na'], time_lim=time_lim, optimizer_name="ASGD")
# print('Best gamma:', gamma)
optimizer = AsynchronousSGD(transport, point, gamma=0.00032, delay_adaptive=True)

print(optimizer.__class__.__name__)

x_0 = f(point) - f(analytical_solution)
iteration_grads = [np.linalg.norm(function.gradient(point)) **2]
iteration_points = [x_0]
iteration_times = [0]


while iteration_times[-1] < time_lim:
    optimizer.step()
    iteration_points.append(f(optimizer.get_point()) - f(analytical_solution))
    iteration_times.append(optimizer.get_time())
    iteration_grads.append(np.linalg.norm(
        function.gradient(optimizer.get_point())) ** 2)


plt.semilogy(iteration_times, iteration_grads, label=r"Delay-Adaptive ASGD: $\gamma=0.00032$", linestyle='solid', marker=markers[1], markersize=16, markevery=max(1, len(iteration_times) // 10), color=colors[1])



# optimizer = RingmasterASGD_adaptive(transport, point, max_delay=6, gamma=0.15)
# print(optimizer.__class__.__name__)

# x_0 = f(point) - f(analytical_solution)
# iteration_grads = [np.linalg.norm(function.gradient(point)) **2]
# iteration_points = [x_0]
# iteration_times = [0]


# while iteration_times[-1] < time_lim:
#     optimizer.step()
#     iteration_times.append(optimizer.get_time())
#     iteration_grads.append(np.linalg.norm(
#         function.gradient(optimizer.get_point())) ** 2)


# plt.semilogy(iteration_times, iteration_grads, label=r"Adaptive Ringmaster ASGD: $\gamma=0.2$, $R=6$", linestyle='solid', color='purple', alpha = 0.6)


plt.legend(loc='upper right', prop={'size': 16})
plt.xlim(0, time_lim)
plt.xlabel("Runtime (seconds)")
plt.ylabel(r"$f(x^t)-f^{\inf}$")
plt.show()