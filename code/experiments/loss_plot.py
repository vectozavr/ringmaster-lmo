import os
import pickle
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

sns.set(style="whitegrid", context="talk", font_scale=1.2, palette=sns.color_palette("bright"), color_codes=False)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['figure.figsize'] = (10, 7)

"""
Put your loss curve here
"""
iterations = 3_000_001  # Total number of iterations
decay_rate = 0.000001      # Rate at which the loss decreases

# Generate the loss values over iterations
iterations_array = np.arange(iterations)
loss_values = np.exp(-decay_rate * iterations_array)

# Plotting the loss curve
# plt.figure(figsize=(10, 6))
# plt.plot(iterations_array, loss_values, label='Loss curve (First 1000 iterations)', color='blue')  # Plot first 1000 iterations for clarity
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.title('Typical Loss Curve Over 3 Million Iterations')
# plt.legend()
# plt.grid(True)
# plt.show()

# Load the file
file_path = os.path.join(os.path.dirname(__file__), "losses_23batch.txt")  # Update this path if needed
# Read the contents
with open(file_path, "r") as file:
    data = file.readlines()
# Extract only the second column (loss values)
loss_values = np.array([float(line.strip().split(",")[1]) for line in data if line.strip()])
iterations = len(loss_values)
print(iterations)


window_size = 2000
local_averages = np.convolve(loss_values, np.ones(window_size)/window_size, mode='valid')
loss_values = local_averages
iterations = len(loss_values)
print(iterations)

"""
Hyperparameters
"""
beta = 29
num_nodes = 153 # 17 51 153 459
# time_setup = f'{beta}log(i+2)+Exp({beta}log(i+2))'
# time_setup = f'{beta}sqrt(i+1)+Exp({beta}sqrt(i+1))'
time_setup = f'{beta}(i+1)+Exp({beta}(i+1))'
# time_setup = f'{beta}(i+1)**2+Exp({beta}(i+1)**2)'
# time_setup = 'Gamma'

batch_size = 23
dim = 100
seed = 5

# Bandit Run Folder
run_setup = f'BANDIT_n={num_nodes}_t={time_setup}_B={batch_size}_dim={dim}_seed={seed}'
            
alg_names = [
    'ATA',
    'ATA: Empirical',
    'FTA: Optimal',
    'GTA',
    'UTA',
    ]
colors = {
    "ATA": "#C00000",  # Cherry Red
    "ATA: Empirical": "#B8860B",  # Dark Golden Orange
    "FTA: Optimal": "#006666",  # Dark Teal
    "GTA": "#36454F",  # Charcoal Gray
    "UTA": "#8B2500"  # Deep Rust
    }
markers = {
        "ATA": "d",
        "ATA: Empirical": "*",
        "FTA: Optimal": "^",
        "GTA": "H",
        "UTA": "v"
    }
alpha = 1
linestyle = 'solid' # solid dashed dashdot

# Create the plot directory if it doesn't exist
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(plot_dir, exist_ok=True)
# Create a directory for the plots using the file name
plot_dir = os.path.join(os.path.dirname(__file__), 'plots', run_setup)
os.makedirs(plot_dir, exist_ok=True)
print(plot_dir)

results = {}

for name in alg_names:
    file_path = os.path.join(os.path.dirname(__file__), 'runs', run_setup, name + '.pkl')
    with open(file_path, 'rb') as f:
        results[name] = pickle.load(f)
        results[name] = results[name][:1] + (loss_values,) + results[name][1:]


# gta_rt = results['GTA'][0][-1]

if '29(i+1)' in run_setup:
    max_runtime = 0.75*1e10
elif 'sqrt(i+1)' in run_setup:
    max_runtime = 2.5*1e9
elif 'log(i+2)' in run_setup:
    if num_nodes==17:
        max_runtime = 3*1e9
    elif num_nodes ==51:
        max_runtime = 3*1e9
    elif num_nodes==153:
        max_runtime = 3*1e9
    elif num_nodes==459:
        max_runtime = 3*1e9
elif 'Gamma' in run_setup:
    max_runtime = 2*1e9
else:
    max_runtime = 2*1e9
for i, (name, (iteration_times, grads, _, _, _, _)) in enumerate(results.items()):
    # if name != 'GTA':
    #     print(f"RT Ratio for {name} = {round(iteration_times[-1]/gta_rt, 2)}")

    iteration_times = np.array(iteration_times)
    # iteration_times = iteration_times[1_000_000:]
    iteration_times = iteration_times[:iterations]
    indicies = np.where(iteration_times <= max_runtime)[0]
    iteration_times = iteration_times[indicies]
    grads = np.array(grads)[indicies]
    plt.plot(
        iteration_times[1:],
        grads[1:],
        label=name,
        linestyle=linestyle,
        alpha=alpha,
        color=colors.get(name, 'black'),
        marker=markers.get(name, 'x'),
        markevery=max(1, int(len(iteration_times) // (25 + i) * max(1, (max_runtime / iteration_times[-1]))))
    )

# plt.yscale('log')
plt.legend(prop={'size': 14})
# plt.xlim(right = max_runtime)
# plt.ylim(top = 1e-1)
plt.xlabel("Runtime")
plt.ylabel(r"$f(x_k)-f^{*}$")
print(1)
plt.savefig(os.path.join(plot_dir, 'plot1_runtime_vs_grad.pdf'))
print(2)
plt.close()
print(3)


# gta_tt = results['GTA'][2][-1]

if '29(i+1)' in run_setup:
    max_time = 0.5*1e11
elif 'sqrt(i+1)' in run_setup:
    max_time =2.5*1e10
elif 'log(i+2)' in run_setup:
    if num_nodes==17:
        max_time = 3*1e10
    elif num_nodes ==51:
        max_time = 3*1e10
    elif num_nodes==153:
        max_time = 3*1e10
    elif num_nodes==459:
        max_time = 3*1e10
elif 'Gamma' in run_setup:
    max_time = 2*1e9
else:
    max_time = max_time = 3*1e10
for i, (name, (_, grads, total_worker_time, _, _, _)) in enumerate(results.items()):
    # if name not in {'GTA', 'UTA'}:
    #     print(f"TT Ratio for {name} = {round(gta_tt/total_worker_time[-1], 2)}")

    total_worker_time = np.array(total_worker_time)
    total_worker_time = total_worker_time[:iterations]
    indicies = np.where(total_worker_time <= max_time)[0]
    total_worker_time = total_worker_time[indicies]
    grads = np.array(grads)[indicies]
    plt.plot(
        total_worker_time[1:],
        grads[1:],
        label=name,
        linestyle='solid',
        alpha=alpha,
        color=colors.get(name, 'black'),
        marker=markers.get(name, 'x'),
        markevery=max(1, int(len(total_worker_time) // (25 + i) * max(1, (max_time / total_worker_time[-1]))))
    )

# plt.yscale('log')
plt.legend(prop={'size': 14})
plt.ylabel(r"$f(x_k)-f^{*}$")
plt.xlabel("Total worker time")
# plt.xlim(right = max_time)
# plt.ylim(top = 1e-1)
print(1)
plt.savefig(os.path.join(plot_dir, 'plot2_total_worker_time_vs_grad.pdf'))
print(2)
plt.close()
