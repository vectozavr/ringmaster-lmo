# run_experiments.py
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from asynchronous.asynchronous_transport import RandomDelayedAsynchronousTransport
from asynchronous.algorithm import StochasticGradientNodeAlgorithm, RingleaderASGD, MaleniaSGD, IAASGD
from signature import Signature
from utils_NN import prepare_dataset
from model import SimpleNeuralNetFunction
from prep_data import dirichlet_partition_equal_size, reduce_dataset_to_multiple

# ---------------------------
# Experiment factories (isolated per run)
# ---------------------------

def make_delays(num_nodes: int, seed: int):
    base_delays = np.array([1 + i for i in range(num_nodes)], dtype=int)
    rng_delays = np.random.default_rng(seed=seed)
    return rng_delays.permutation(base_delays)

def make_noise_function(num_nodes: int, seed: int):
    gens = [np.random.default_rng(seed + i) for i in range(num_nodes)]
    def halfnormal(index):
        return abs(gens[index].normal(loc=0.0, scale=np.sqrt(index + 1)))
    return halfnormal

def make_transport(client_X, client_y, delays, seed):
    nodes = [Signature(StochasticGradientNodeAlgorithm,
                       SimpleNeuralNetFunction(X, y, seed=seed))
             for (X, y) in zip(client_X, client_y)]
    noise_fn = make_noise_function(num_nodes=len(nodes), seed=seed)
    return RandomDelayedAsynchronousTransport(nodes, delays.copy(), noise_fn)

def run_one_method(method_name, gamma, features, labels, client_X, client_y, delays, time_lim, seed):
    """
    Run a single optimizer and return traces of time, grad-norm^2, and training loss.
    """
    # fresh function & initial point
    function  = SimpleNeuralNetFunction(features, labels, seed=seed)
    init_point = np.array(function.get_current_point(), copy=True)

    # fresh transport (nodes + noise)
    transport = make_transport(client_X, client_y, delays, seed)

    # optimizer
    if method_name == "ringleader":
        Opt = RingleaderASGD
    elif method_name == "malenia":
        Opt = MaleniaSGD
    elif method_name == "iaasgd":
        Opt = IAASGD
    else:
        raise ValueError(f"Unknown method {method_name}")

    optimizer = Opt(transport, init_point, gamma=gamma)

    # trace
    times  = [0.0]
    grads2 = [np.linalg.norm(function.gradient(init_point))**2]
    losses = [float(function.value(init_point))]

    while times[-1] < time_lim:
        optimizer.step()
        times.append(optimizer.get_time())
        # record metrics at the current optimizer point
        point = optimizer.get_point()
        grads2.append(np.linalg.norm(function.gradient(point))**2)
        losses.append(float(function.value(point)))

    return np.array(times), np.array(grads2), np.array(losses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="runs", help="Directory to save runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[26, 27, 28, 29, 30])
    parser.add_argument("--num_nodes", type=int, default=100) # 100
    parser.add_argument("--time_lim", type=float, default=400000) # 300000
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--dataset_root", type=str, default="./")
    parser.add_argument("--subset", type=int, default=None, help="Use first N examples (or None for full train)")
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashion"], default="fashion")

    # Gammas for each method (change if you want)
    parser.add_argument("--gamma_ringleader", type=float, default=0.05)
    parser.add_argument("--gamma_malenia", type=float, default=2.0)
    parser.add_argument("--gamma_iaasgd", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # deterministic dataset load (uses your unified prepare_dataset)
    features, labels, number_of_classes = prepare_dataset(
        args.dataset_root,
        number_of_first_examples=args.subset,
        dataset=args.dataset
    )

    # ensure divisibility across nodes, then make a Dirichlet partition
    features, labels, kept_idx, dropped_idx = reduce_dataset_to_multiple(
        features, labels, number_of_classes, args.num_nodes, seed=123456  # fixed seed; not the sweep seed
    )
    client_X, client_y, client_idxs = dirichlet_partition_equal_size(
        features, labels, number_of_classes,
        num_nodes=args.num_nodes, alpha=args.alpha, seed=654321  # fixed for all seeds
    )

    # methods config
    methods = [
        ("ringleader", args.gamma_ringleader),
        ("malenia",    args.gamma_malenia),
        ("iaasgd",     args.gamma_iaasgd),
    ]

    # run across seeds
    for seed in args.seeds:
        print(f"\n=== Running seed {seed} ===")
        delays = make_delays(args.num_nodes, seed)  # same within seed for all methods

        for method_name, gamma in methods:
            print(f" -> {method_name}, gamma={gamma}")
            times, grads2, losses = run_one_method(
                method_name, gamma,
                features, labels, client_X, client_y, delays,
                time_lim=args.time_lim, seed=seed
            )

            # save as .npz (compact & safe)
            fname = f"{method_name}_seed{seed}.npz"
            path = os.path.join(args.output_dir, fname)
            np.savez_compressed(
                path,
                times=times,
                grads=grads2,      # gradient norm squared
                loss=losses,       # training loss
                meta=json.dumps({
                    "method": method_name,
                    "gamma": gamma,
                    "seed": seed,
                    "num_nodes": args.num_nodes,
                    "time_lim": args.time_lim,
                    "alpha": args.alpha,
                    "subset": args.subset,
                    "dataset": args.dataset,
                })
            )
            print(f"    saved: {path}")

if __name__ == "__main__":
    main()
