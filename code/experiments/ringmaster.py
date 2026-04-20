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
    DelayAdaptiveMuonASGD,
    ParameterAgnosticRingmasterMuonASGD,
    RennalaMuonSGD,
    RingmasterMuonASGD,
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


colors = {
    "ringmaster_muon": "#C00000",
    "pa_ringmaster_muon": "#B8860B",
    "rennala_muon": "#006666",
    "delay_adaptive_muon": "#4682B4",
}

markers = {
    "ringmaster_muon": "*",
    "pa_ringmaster_muon": "^",
    "rennala_muon": "H",
    "delay_adaptive_muon": "o",
}


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
            "Ringmaster Muon",
            RingmasterMuonASGD(
                build_transport(function, num_nodes, seed=11),
                point.copy(),
                max_delay=int(os.environ.get("RINGMASTER_MUON_MAX_DELAY", "4")),
                gamma=float(os.environ.get("RINGMASTER_MUON_GAMMA", "0.004")),
                beta=float(os.environ.get("MUON_BETA", "0.95")),
                ns_steps=int(os.environ.get("MUON_NS_STEPS", "5")),
                nesterov=os.environ.get("MUON_NESTEROV", "1") != "0",
                meta=muon_meta,
            ),
            dict(
                color=colors["ringmaster_muon"],
                marker=markers["ringmaster_muon"],
                markersize=18,
                label=(
                    f"Ringmaster Muon: "
                    f"gamma={os.environ.get('RINGMASTER_MUON_GAMMA', '0.004')}, "
                    f"R={os.environ.get('RINGMASTER_MUON_MAX_DELAY', '4')}"
                ),
            ),
        ),
        (
            "PA Ringmaster Muon",
            ParameterAgnosticRingmasterMuonASGD(
                build_transport(function, num_nodes, seed=13),
                point.copy(),
                eta=float(os.environ.get("PA_RINGMASTER_MUON_ETA", "0.05")),
                ns_steps=int(os.environ.get("MUON_NS_STEPS", "5")),
                nesterov=os.environ.get("MUON_NESTEROV", "1") != "0",
                meta=muon_meta,
            ),
            dict(
                color=colors["pa_ringmaster_muon"],
                marker=markers["pa_ringmaster_muon"],
                markersize=16,
                label=f"PA Ringmaster Muon: eta={os.environ.get('PA_RINGMASTER_MUON_ETA', '0.05')}",
            ),
        ),
        (
            "Rennala Muon",
            RennalaMuonSGD(
                build_transport(function, num_nodes, seed=17),
                point.copy(),
                gamma=float(os.environ.get("RENNALA_MUON_GAMMA", "0.004")),
                batch_size=int(os.environ.get("RENNALA_MUON_BATCH_SIZE", "4")),
                beta=float(os.environ.get("MUON_BETA", "0.95")),
                ns_steps=int(os.environ.get("MUON_NS_STEPS", "5")),
                nesterov=os.environ.get("MUON_NESTEROV", "1") != "0",
                meta=muon_meta,
            ),
            dict(
                color=colors["rennala_muon"],
                marker=markers["rennala_muon"],
                markersize=16,
                label=(
                    f"Rennala Muon: "
                    f"gamma={os.environ.get('RENNALA_MUON_GAMMA', '0.004')}, "
                    f"B={os.environ.get('RENNALA_MUON_BATCH_SIZE', '4')}"
                ),
            ),
        ),
        (
            "Delay-Adaptive Muon",
            DelayAdaptiveMuonASGD(
                build_transport(function, num_nodes, seed=19),
                point.copy(),
                gamma=float(os.environ.get("DELAY_ADAPTIVE_MUON_GAMMA", "0.0002")),
                delay_adaptive=True,
                beta=float(os.environ.get("MUON_BETA", "0.95")),
                ns_steps=int(os.environ.get("MUON_NS_STEPS", "5")),
                nesterov=os.environ.get("MUON_NESTEROV", "1") != "0",
                meta=muon_meta,
            ),
            dict(
                color=colors["delay_adaptive_muon"],
                marker=markers["delay_adaptive_muon"],
                markersize=14,
                label=f"Delay-Adaptive Muon: gamma={os.environ.get('DELAY_ADAPTIVE_MUON_GAMMA', '0.0002')}",
            ),
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
    plt.title(f"Nanochat Muon Async Training Comparison (depth={DEPTH})")

    if os.environ.get("DISPLAY"):
        plt.show()
    else:
        output_path = os.path.join(experiment_dir, "ringmaster_nanochat.png")
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
