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
Hyperparameters
"""
beta = 29

num_nodes = 50
time_setup = f'{beta}(5*g+1)+Hetero({beta}(5*g+1))'

batch_size = 23
dim = 100
gamma = 0.0001
seed = 5

# Run Folder
run_setup = f'n={num_nodes}_t={time_setup}_B={batch_size}_dim={dim}_gamma={gamma}_seed={seed}'

alg_names = [
    'ATA',
    'ATA: Empirical',
    'FTA: Optimal',
    'GTA',
    'UTA'
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

# results = {name: [] for name in alg_names}
# for seed in seeds:
#     run = run_setup + f'_seed={seed}'
#     for name in alg_names:
#         file_path = os.path.join(os.path.dirname(__file__), 'runs', run, name + '.pkl')
#         try:
#             with open(file_path, 'rb') as f:
#                 results[name].append(pickle.load(f))
#         except FileNotFoundError:
#             print(f"File not found: {file_path}")


# for name, result_list in results.items():
#     # Initialize arrays to store aggregated results for each component
#     iteration_times_all_seeds = []
#     grads_all_seeds = []
#     total_worker_time_all_seeds = []
#     allocations_all_seeds = []
#     warm_start_all_seeds = []
#     expected_iteration_time_all_seeds = []
    
#     for result in result_list:
#         iteration_times, grads, total_worker_time, allocations, warm_start, expected_iteration_time = result
        
#         # Append the results from this seed to the corresponding lists
#         iteration_times_all_seeds.append(iteration_times)
#         grads_all_seeds.append(grads)
#         total_worker_time_all_seeds.append(total_worker_time)
#         # allocations_all_seeds.append(allocations)
#         # if warm_start is not None:
#             # warm_start_all_seeds.append(warm_start)
#         expected_iteration_time_all_seeds.append(expected_iteration_time)
    
#     # Convert lists to numpy arrays for easier manipulation
#     iteration_times_all_seeds = np.array(iteration_times_all_seeds)
#     grads_all_seeds = np.array(grads_all_seeds)
#     total_worker_time_all_seeds = np.array(total_worker_time_all_seeds)
#     # allocations_all_seeds = np.array(allocations_all_seeds)
#     # warm_start_all_seeds = np.array(warm_start_all_seeds) if warm_start_all_seeds else None
#     expected_iteration_time_all_seeds = np.array(expected_iteration_time_all_seeds)
    
#     # Update the results dictionary with aggregated arrays
#     results[name] = (
#         iteration_times_all_seeds,
#         grads_all_seeds,
#         total_worker_time_all_seeds,
#         None,
#         None,
#         expected_iteration_time_all_seeds
#     )

gta_rt = results['GTA'][0][-1]

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
    if name != 'GTA':
        print(f"RT Ratio for {name} = {round(iteration_times[-1]/gta_rt, 2)}")

    iteration_times = np.array(iteration_times)
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
        markevery=max(1, int(len(iteration_times) // (10 + i) * max(1, (max_runtime / iteration_times[-1]))))
    )

plt.yscale('log')
plt.legend(prop={'size': 14})
plt.xlim(right = max_runtime)
plt.ylim(top = 1e-1)
plt.xlabel("Runtime")
plt.ylabel(r"$f(x_k)-f^{*}$")
print(1)
plt.savefig(os.path.join(plot_dir, 'plot1_runtime_vs_grad.pdf'))
print(2)
plt.close()
print(3)


gta_tt = results['GTA'][2][-1]

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
    if name not in {'GTA', 'UTA'}:
        print(f"TT Ratio for {name} = {round(gta_tt/total_worker_time[-1], 2)}")

    total_worker_time = np.array(total_worker_time)
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
        markevery=max(1, int(len(total_worker_time) // (10) * max(1, (max_time / total_worker_time[-1]))))
    )

plt.yscale('log')
plt.legend(prop={'size': 14})
plt.ylabel(r"$f(x_k)-f^{*}$")
plt.xlabel("Total worker time")
plt.xlim(right = max_time)
plt.ylim(top = 1e-1)
print(1)
plt.savefig(os.path.join(plot_dir, 'plot2_total_worker_time_vs_grad.pdf'))
print(2)
plt.close()


for i, (name, (iteration_times, _, _, _, _, _)) in enumerate(results.items()):
    plt.plot(
        np.divide(iteration_times[1:], np.arange(1, len(iteration_times[1:]) + 1)),
        label=name,
        linestyle='solid',
        alpha=alpha,
        color=colors.get(name, 'black'),
        marker=markers.get(name, 'x'),
        markevery=max(1, len(iteration_times) // (10 + i))
    )
plt.legend(prop={'size': 14})
plt.xlabel('Iterations')
plt.ylabel("Average iteration time")
plt.savefig(os.path.join(plot_dir, 'plot3_iterations_vs_average_iteration_time.pdf'))
plt.close()

(_, _, _, _, _, optimal_expected_iteration_time) = results['FTA: Optimal']

for i, (name, (_, _, _, _, _, expected_iteration_time)) in enumerate(results.items()):
    if name in {'GTA', 'FTA: Optimal'}:
        continue
    plt.plot(
        np.divide(np.subtract(expected_iteration_time[1:], optimal_expected_iteration_time[1:]), np.arange(1,len(expected_iteration_time[1:])+1)),
        label=name,
        linestyle='solid',
        alpha=alpha,
        color=colors.get(name, 'black'),
        marker=markers.get(name, 'x'),
        markevery=max(1, len(expected_iteration_time) // (10 + i))
    )
plt.legend(prop={'size': 14})
plt.xlabel("Iterations")
plt.ylabel("Average regret")
# plt.yscale('log')
plt.savefig(os.path.join(plot_dir, 'plot4_iterations_vs_proxy_avg_regret.pdf'))
plt.close()

print(plot_dir)