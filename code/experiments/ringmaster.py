import argparse
import gc
import itertools
import json
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


# Experiment defaults. Edit these directly for the standard Nanochat run.
DEFAULT_NUM_NODES = 8
DEFAULT_TIME_LIMIT = 300.0
DEFAULT_EVAL_INTERVAL = 5.0
DEFAULT_DEVICE_BATCH_SIZE = 2
DEFAULT_EVAL_BATCH_SIZE = 4
DEFAULT_NUM_SHARDS = 10

FUNCTION_SEED = 7
TRANSPORT_SEED = 11

COMMON_MUON_PARAMS = {
    "beta": 0.95,
    "ns_steps": 5,
    "nesterov": True,
}

DEFAULT_PARAMS = {
    "RingmasterMuonASGD": {
        "gamma": 0.004,
        "max_delay": 4,
    },
    "ParameterAgnosticRingmasterMuonASGD": {
        "eta": 0.05,
    },
    "RennalaMuonSGD": {
        "gamma": 0.004,
        "batch_size": 4,
    },
    "DelayAdaptiveMuonASGD": {
        "gamma": 0.0002,
    },
}

TUNING_GRIDS = {
    "RingmasterMuonASGD": {
        "gamma": [5 ** i for i in range(-6, 1)],
        "max_delay": [1, 2, 4, 8, 16],
    },
    "ParameterAgnosticRingmasterMuonASGD": {
        "eta": [5 ** i for i in range(-5, 2)],
    },
    "RennalaMuonSGD": {
        "gamma": [5 ** i for i in range(-6, 1)],
        "batch_size": [1, 2, 4, 8, 16],
    },
    "DelayAdaptiveMuonASGD": {
        "gamma": [5 ** i for i in range(-8, -1)],
    },
}

COLORS = {
    "RingmasterMuonASGD": "#C00000",
    "ParameterAgnosticRingmasterMuonASGD": "#B8860B",
    "RennalaMuonSGD": "#006666",
    "DelayAdaptiveMuonASGD": "#4682B4",
}

MARKERS = {
    "RingmasterMuonASGD": "*",
    "ParameterAgnosticRingmasterMuonASGD": "^",
    "RennalaMuonSGD": "H",
    "DelayAdaptiveMuonASGD": "o",
}

MARKER_SIZES = {
    "RingmasterMuonASGD": 18,
    "ParameterAgnosticRingmasterMuonASGD": 16,
    "RennalaMuonSGD": 16,
    "DelayAdaptiveMuonASGD": 14,
}

PLOTTING_ORDER = [
    "RingmasterMuonASGD",
    "ParameterAgnosticRingmasterMuonASGD",
    "RennalaMuonSGD",
    "DelayAdaptiveMuonASGD",
]


def clone_point(point):
    if isinstance(point, torch.Tensor):
        return point.clone()
    return point.copy()


def build_function(function_seed):
    return NanochatLanguageModelFunction(
        seed=function_seed,
        device_batch_size=DEFAULT_DEVICE_BATCH_SIZE,
        eval_batch_size=DEFAULT_EVAL_BATCH_SIZE,
        num_shards=DEFAULT_NUM_SHARDS,
        is_cuda=torch.cuda.is_available(),
    )


def build_transport(function, num_nodes, seed):
    delays = np.array([1.0 + np.sqrt(i + 1) for i in range(num_nodes)], dtype=np.float64)
    generator = np.random.default_rng(seed=seed)

    def halfnormal(index):
        return np.abs(generator.normal(loc=0.0, scale=np.sqrt(index + 1) * 0.05))

    nodes = [Signature(StochasticGradientNodeAlgorithm, function) for _ in range(num_nodes)]
    return RandomDelayedAsynchronousTransport(nodes, delays, halfnormal)


def merge_optimizer_params(method_name, params):
    merged = dict(params)
    if method_name != "ParameterAgnosticRingmasterMuonASGD":
        for key, value in COMMON_MUON_PARAMS.items():
            merged.setdefault(key, value)
    else:
        merged.setdefault("ns_steps", COMMON_MUON_PARAMS["ns_steps"])
        merged.setdefault("nesterov", COMMON_MUON_PARAMS["nesterov"])
    return merged


def build_optimizer(method_name, params, point, transport, muon_meta):
    params = merge_optimizer_params(method_name, params)
    if method_name == "RingmasterMuonASGD":
        return RingmasterMuonASGD(
            transport,
            clone_point(point),
            max_delay=params["max_delay"],
            gamma=params["gamma"],
            beta=params["beta"],
            ns_steps=params["ns_steps"],
            nesterov=params["nesterov"],
            meta=muon_meta,
        )
    if method_name == "ParameterAgnosticRingmasterMuonASGD":
        return ParameterAgnosticRingmasterMuonASGD(
            transport,
            clone_point(point),
            eta=params["eta"],
            ns_steps=params["ns_steps"],
            nesterov=params["nesterov"],
            meta=muon_meta,
        )
    if method_name == "RennalaMuonSGD":
        return RennalaMuonSGD(
            transport,
            clone_point(point),
            gamma=params["gamma"],
            batch_size=params["batch_size"],
            beta=params["beta"],
            ns_steps=params["ns_steps"],
            nesterov=params["nesterov"],
            meta=muon_meta,
        )
    if method_name == "DelayAdaptiveMuonASGD":
        return DelayAdaptiveMuonASGD(
            transport,
            clone_point(point),
            gamma=params["gamma"],
            delay_adaptive=True,
            beta=params["beta"],
            ns_steps=params["ns_steps"],
            nesterov=params["nesterov"],
            meta=muon_meta,
        )
    raise ValueError(f"Unknown method {method_name}")


def run_optimizer(optimizer, function, point, time_lim, eval_interval):
    runtime = [0.0]
    latest_train_loss = [function.value(point)]
    next_eval_time = eval_interval

    while runtime[-1] < time_lim:
        optimizer.step()
        current_time = optimizer.get_time()
        if current_time >= next_eval_time or current_time >= time_lim:
            runtime.append(min(current_time, time_lim))
            latest_train_loss.append(function.value(optimizer.get_point()))
            next_eval_time += eval_interval

    return {
        "runtime": runtime,
        "latest_train_loss": latest_train_loss,
    }


def evaluate_method(method_name, params, time_lim, eval_interval, function_seed, transport_seed):
    function = build_function(function_seed)
    point = function.get_current_point()
    muon_meta = function.parameter_metadata()
    transport = build_transport(function, DEFAULT_NUM_NODES, transport_seed)
    optimizer = build_optimizer(method_name, params, point, transport, muon_meta)
    trace = run_optimizer(optimizer, function, point, time_lim, eval_interval)
    del optimizer, transport, function
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return trace


def score_trace(trace):
    return float(trace["latest_train_loss"][-1])


def candidate_param_sets(method_name):
    grid = TUNING_GRIDS[method_name]
    keys = list(grid.keys())
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def tuning_plot_path(method_name):
    method_slug = format_method_name(method_name).lower().replace("-", "").replace(" ", "_")
    return os.path.join(os.path.dirname(__file__), f"tuning_{method_slug}_nanochat.pdf")


def format_params_for_tuning_label(params):
    return ", ".join(f"{name}={value}" for name, value in params.items())


def format_method_name(method_name):
    if method_name == "RingmasterMuonASGD":
        return "Ringmaster Muon"
    if method_name == "ParameterAgnosticRingmasterMuonASGD":
        return "Parameter-Agnostic Ringmaster Muon"
    if method_name == "RennalaMuonSGD":
        return "Rennala Muon"
    if method_name == "DelayAdaptiveMuonASGD":
        return "Delay-Adaptive Muon"
    raise ValueError(f"Unknown method {method_name}")


def plot_tuning_lines_for_method(method_name, time_lim, traces_by_params):
    plt.figure()
    for params, trace in traces_by_params:
        plt.semilogy(
            trace["runtime"],
            trace["latest_train_loss"],
            label=format_params_for_tuning_label(params),
            linestyle="solid",
            alpha=0.85,
        )

    plt.legend(loc="best", prop={"size": 10})
    plt.xlim(0, time_lim)
    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Latest Minibatch Loss")
    plt.title(f"Tuning {format_method_name(method_name)} method")
    plt.tight_layout()
    output_path = tuning_plot_path(method_name)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved tuning plot to {output_path}")


def tune_method(method_name, time_lim, num_trials, aggregator, eval_interval, plot_lines=False):
    best_score = np.inf
    best_params = None
    best_trial_scores = None
    traces_by_params = []

    for params in candidate_param_sets(method_name):
        trial_scores = []
        for trial_index in range(num_trials):
            trace = evaluate_method(
                method_name,
                params,
                time_lim,
                eval_interval,
                function_seed=FUNCTION_SEED + trial_index,
                transport_seed=TRANSPORT_SEED + trial_index,
            )
            if plot_lines and trial_index == 0:
                traces_by_params.append((dict(params), trace))
            trial_scores.append(score_trace(trace))

        aggregate_score = float(np.mean(trial_scores) if aggregator == "mean" else np.median(trial_scores))
        print(
            f"{method_name}: score={aggregate_score:.6f}, "
            f"trial_scores={[float(score) for score in trial_scores]}, params={params}"
        )

        if aggregate_score < best_score:
            best_score = aggregate_score
            best_params = dict(params)
            best_trial_scores = [float(score) for score in trial_scores]

    if plot_lines:
        plot_tuning_lines_for_method(method_name, time_lim, traces_by_params)

    return {
        "params": best_params,
        "score": float(best_score),
        "metric": "latest_train_loss",
        "num_trials": num_trials,
        "aggregator": aggregator,
        "trial_scores": best_trial_scores,
    }


def tune_all_methods(time_lim, num_trials, aggregator, eval_interval, plot_lines=False):
    tuned = {}
    for method_name in PLOTTING_ORDER:
        print(f"Tuning {method_name}...")
        tuned[method_name] = tune_method(
            method_name,
            time_lim,
            num_trials,
            aggregator,
            eval_interval,
            plot_lines=plot_lines,
        )
    return tuned


def save_tuned_params(tuned_results, output_path, time_lim, num_trials, aggregator):
    payload = {
        "_meta": {
            "time_limit": time_lim,
            "metric": "latest_train_loss",
            "num_trials": num_trials,
            "aggregator": aggregator,
            "common_muon_params": COMMON_MUON_PARAMS,
            "num_nodes": DEFAULT_NUM_NODES,
            "device_batch_size": DEFAULT_DEVICE_BATCH_SIZE,
            "eval_batch_size": DEFAULT_EVAL_BATCH_SIZE,
            "num_shards": DEFAULT_NUM_SHARDS,
        }
    }
    payload.update(tuned_results)
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)


def load_params(params_path):
    with open(params_path, "r", encoding="utf-8") as source:
        loaded = json.load(source)

    params = {}
    for method_name, defaults in DEFAULT_PARAMS.items():
        if method_name not in loaded:
            raise ValueError(f"Missing parameters for {method_name} in {params_path}")
        if isinstance(loaded[method_name], dict) and "params" in loaded[method_name]:
            params[method_name] = {**defaults, **loaded[method_name]["params"]}
        else:
            params[method_name] = {**defaults, **loaded[method_name]}
    return params


def format_method_label(method_name, params):
    if method_name == "RingmasterMuonASGD":
        return f"Ringmaster Muon: gamma={params['gamma']}, R={params['max_delay']}"
    if method_name == "ParameterAgnosticRingmasterMuonASGD":
        return f"PA Ringmaster Muon: eta={params['eta']}"
    if method_name == "RennalaMuonSGD":
        return f"Rennala Muon: gamma={params['gamma']}, B={params['batch_size']}"
    if method_name == "DelayAdaptiveMuonASGD":
        return f"Delay-Adaptive Muon: gamma={params['gamma']}"
    raise ValueError(f"Unknown method {method_name}")


def plot_comparison(params_by_method, time_lim, eval_interval, title_suffix="", trial_index=0):
    plt.figure()

    for method_name in PLOTTING_ORDER:
        trace = evaluate_method(
            method_name,
            params_by_method[method_name],
            time_lim,
            eval_interval,
            function_seed=FUNCTION_SEED + trial_index,
            transport_seed=TRANSPORT_SEED + trial_index,
        )
        print(f"Comparing {method_name} with params={params_by_method[method_name]}")
        plt.semilogy(
            trace["runtime"],
            trace["latest_train_loss"],
            label=format_method_label(method_name, params_by_method[method_name]),
            linestyle="solid",
            marker=MARKERS[method_name],
            markersize=MARKER_SIZES[method_name],
            markevery=max(1, len(trace["runtime"]) // 10),
            color=COLORS[method_name],
        )

    plt.legend(loc="upper right", prop={"size": 14})
    plt.xlim(0, time_lim)
    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Latest Minibatch Loss")
    title = f"Nanochat Muon Async Training Comparison (depth={DEPTH})"
    if title_suffix:
        title = f"{title}: {title_suffix}"
    plt.title(title)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), "ringmaster_nanochat.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def print_tuning_summary():
    print("Hyperparameters tuned by default:")
    print("  Ringmaster Muon: gamma and max_delay (refresh threshold R).")
    print("  PA Ringmaster Muon: eta only; alpha_k, threshold R_k, and Muon momentum schedule follow the theorem.")
    print("  Rennala Muon: gamma and batch_size B.")
    print("  Delay-Adaptive Muon: gamma only.")
    print("Shared Muon internals kept fixed by default:")
    print(
        f"  beta={COMMON_MUON_PARAMS['beta']}, "
        f"ns_steps={COMMON_MUON_PARAMS['ns_steps']}, "
        f"nesterov={COMMON_MUON_PARAMS['nesterov']}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Tune and compare Muon-based asynchronous Nanochat methods. "
            "Use --mode tune_and_compare to search hyperparameters first and then plot the methods, "
            "or use compare/compare_defaults to plot without tuning in the current run."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["tune", "compare", "compare_defaults", "tune_and_compare"],
        default="compare",
        help=(
            "`tune` searches hyperparameters, "
            "`compare` plots using a parameter set, "
            "`compare_defaults` plots using the hard-coded defaults, "
            "`tune_and_compare` does both."
        ),
    )
    parser.add_argument(
        "--params-file",
        default=os.path.join(os.path.dirname(__file__), "ringmaster_nanochat_muon_tuned_params.json"),
        help="JSON file used to save tuned params to or load params from.",
    )
    parser.add_argument(
        "--time-lim",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help="Runtime horizon used for the final comparison plot.",
    )
    parser.add_argument(
        "--tuning-time-lim",
        type=float,
        default=None,
        help="Optional shorter runtime horizon used only during hyperparameter tuning. Defaults to --time-lim.",
    )
    parser.add_argument(
        "--eval-interval",
        type=float,
        default=DEFAULT_EVAL_INTERVAL,
        help="How often to evaluate validation BPB during tuning and comparison.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of independent trials used to score each candidate during tuning.",
    )
    parser.add_argument(
        "--aggregator",
        choices=["mean", "median"],
        default="mean",
        help="How to aggregate tuning scores across trials.",
    )
    parser.add_argument(
        "--plot-tuning-lines",
        action="store_true",
        help="Save one PDF per method with all tuning curves from that method's hyperparameter grid.",
    )
    parser.add_argument(
        "--use-default-params",
        action="store_true",
        help="In compare mode, ignore the params file and use the hard-coded defaults in this script.",
    )
    parser.add_argument(
        "--compare-trial-index",
        type=int,
        default=0,
        help="Which deterministic trial seed to use for the final comparison plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print_tuning_summary()
    sample_function = build_function(FUNCTION_SEED)
    print(f"Nanochat config: {sample_function.parameter_metadata()['model_config']}")
    del sample_function
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tuning_time_lim = args.time_lim if args.tuning_time_lim is None else args.tuning_time_lim

    if args.mode in ("tune", "tune_and_compare"):
        tuned_results = tune_all_methods(
            tuning_time_lim,
            args.num_trials,
            args.aggregator,
            args.eval_interval,
            plot_lines=args.plot_tuning_lines,
        )
        save_tuned_params(
            tuned_results,
            args.params_file,
            tuning_time_lim,
            args.num_trials,
            args.aggregator,
        )
        print(f"Saved tuned parameters to {args.params_file}")
        if args.mode == "tune":
            return
        params_by_method = {method_name: result["params"] for method_name, result in tuned_results.items()}
        plot_comparison(
            params_by_method,
            args.time_lim,
            args.eval_interval,
            title_suffix="tuned",
            trial_index=args.compare_trial_index,
        )
        return

    if args.mode == "compare_defaults" or args.use_default_params:
        params_by_method = DEFAULT_PARAMS
        title_suffix = "defaults"
    else:
        params_by_method = load_params(args.params_file)
        title_suffix = os.path.basename(args.params_file)

    plot_comparison(
        params_by_method,
        args.time_lim,
        args.eval_interval,
        title_suffix=title_suffix,
        trial_index=args.compare_trial_index,
    )


if __name__ == "__main__":
    main()
