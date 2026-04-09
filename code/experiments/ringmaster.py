import os

import matplotlib

if os.environ.get("DISPLAY"):
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from asynchronous.algorithm import (
    AsynchronousSGD,
    RingmasterASGD,
    RingmasterMuonASGD,
    RennalaSGD,
    StochasticGradientNodeAlgorithm,
)
from asynchronous.asynchronous_transport import RandomDelayedAsynchronousTransport
from nanochat_async import DEPTH, NanochatLanguageModelFunction
from signature import Signature


sns.set(style="whitegrid", context="talk", font_scale=1.1, palette=sns.color_palette("bright"), color_codes=False)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "DejaVu Sans"
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["figure.figsize"] = (10, 7)
matplotlib.rcParams["text.usetex"] = False


colors = [
    "#C00000",
    "#B8860B",
    "#006666",
    "#2F5D50",
]

markers = [
    "*",
    "^",
    "H",
    "o",
]


def run_optimizer(optimizer, function, point, time_lim, eval_interval):
    iteration_times = [0.0]
    iteration_losses = [function.value(point)]
    next_eval_time = eval_interval

    while iteration_times[-1] < time_lim:
        optimizer.step()
        current_time = optimizer.get_time()
        if current_time >= next_eval_time or current_time >= time_lim:
            iteration_times.append(min(current_time, time_lim))
            iteration_losses.append(function.value(optimizer.get_point()))
            next_eval_time += eval_interval

    return iteration_times, iteration_losses


def build_transport(function, num_nodes, seed):
    delays = np.array([1.0 + np.sqrt(i + 1) for i in range(num_nodes)], dtype=np.float64)
    generator = np.random.default_rng(seed=seed)

    def halfnormal(index):
        return np.abs(generator.normal(loc=0.0, scale=np.sqrt(index + 1) * 0.05))

    nodes = [Signature(StochasticGradientNodeAlgorithm, function) for _ in range(num_nodes)]
    transport = RandomDelayedAsynchronousTransport(nodes, delays, halfnormal)
    return transport


def main():
    experiment_dir = os.path.dirname(__file__)
    num_nodes = int(os.environ.get("RINGMASTER_NUM_NODES", "8"))
    time_lim = float(os.environ.get("RINGMASTER_TIME_LIM", "300"))
    eval_interval = float(os.environ.get("RINGMASTER_EVAL_INTERVAL", "20"))
    seed = 7

    function = NanochatLanguageModelFunction(
        seed=seed,
        device_batch_size=int(os.environ.get("RINGMASTER_DEVICE_BATCH_SIZE", "2")),
        eval_batch_size=int(os.environ.get("RINGMASTER_EVAL_BATCH_SIZE", "2")),
        num_shards=int(os.environ.get("AUTORESEARCH_NUM_SHARDS", "2")),
        is_cuda=torch.cuda.is_available(),
    )
    point = function.get_current_point()
    muon_meta = function.parameter_metadata()
    print(f"Nanochat config: {muon_meta['model_config']}")

    experiments = [
        (
            "Ringmaster ASGD",
            RingmasterASGD(
                build_transport(function, num_nodes, seed=11),
                point.copy(),
                max_delay=4,
                gamma=float(os.environ.get("RINGMASTER_ASGD_GAMMA", "0.001")),
            ),
            dict(color=colors[0], marker=markers[0], markersize=18, label="Ringmaster ASGD"),
        ),
        (
            "Ringmaster Muon",
            RingmasterMuonASGD(
                build_transport(function, num_nodes, seed=13),
                point.copy(),
                max_delay=4,
                gamma=float(os.environ.get("RINGMASTER_MUON_GAMMA", "0.004")),
                beta=float(os.environ.get("RINGMASTER_MUON_BETA", "0.95")),
                ns_steps=5,
                meta=muon_meta,
            ),
            dict(color=colors[3], marker=markers[3], markersize=14, label="Ringmaster Muon"),
        ),
        (
            "Rennala SGD",
            RennalaSGD(
                build_transport(function, num_nodes, seed=17),
                point.copy(),
                gamma=float(os.environ.get("RENNALA_GAMMA", "0.002")),
                batch_size=4,
            ),
            dict(color=colors[2], marker=markers[2], markersize=16, label="Rennala SGD"),
        ),
        (
            "Delay-Adaptive ASGD",
            AsynchronousSGD(
                build_transport(function, num_nodes, seed=19),
                point.copy(),
                gamma=float(os.environ.get("DELAY_ADAPTIVE_GAMMA", "0.0002")),
                delay_adaptive=True,
            ),
            dict(color=colors[1], marker=markers[1], markersize=16, label="Delay-Adaptive ASGD"),
        ),
    ]

    for name, optimizer, plot_kwargs in experiments:
        print(name)
        iteration_times, iteration_losses = run_optimizer(
            optimizer=optimizer,
            function=function,
            point=point,
            time_lim=time_lim,
            eval_interval=eval_interval,
        )
        plt.semilogy(
            iteration_times,
            iteration_losses,
            linestyle="solid",
            markevery=max(1, len(iteration_times) // 10),
            **plot_kwargs,
        )

    plt.legend(loc="upper right", prop={"size": 14})
    plt.xlim(0, time_lim)
    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Validation BPB")
    plt.title(f"Nanochat Async Training Comparison (depth={DEPTH})")

    if os.environ.get("DISPLAY"):
        plt.show()
    else:
        output_path = os.path.join(experiment_dir, "ringmaster_nanochat.png")
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
