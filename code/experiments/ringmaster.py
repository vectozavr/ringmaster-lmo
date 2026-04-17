import argparse
import itertools
import json
import os

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from asynchronous.algorithm import (
    DelayAdaptiveMuonASGD,
    ParameterAgnosticRingmasterMuonASGD,
    RennalaMuonSGD,
    RingmasterMuonASGD,
    StochasticGradientNodeAlgorithm,
)
from asynchronous.asynchronous_transport import RandomDelayedAsynchronousTransport
from function import StochasticTridiagonalQuadraticFunction
from function import create_worst_case
from signature import Signature


sns.set(style="whitegrid", context="talk", font_scale=1.2, palette=sns.color_palette("bright"), color_codes=False)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "DejaVu Sans"
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["figure.figsize"] = (10, 7)
matplotlib.rcParams["text.usetex"] = True


DIM = 1729
NUM_NODES = 6174
PROBLEM_SEED = 26
TRANSPORT_SEED = 5
NOISE = 0.01
DEFAULT_TIME_LIMIT = 2 * 1e3

DELAYS = np.array([1 + np.sqrt(i) for i in range(NUM_NODES)])

COMMON_MUON_PARAMS = {
    "beta": 0.95,
    "ns_steps": 5,
    "nesterov": True,
}

DEFAULT_PARAMS = {
    "RingmasterMuonASGD": {"gamma": 0.04, "max_delay": 6},
    "ParameterAgnosticRingmasterMuonASGD": {"eta": 0.5},
    "RennalaMuonSGD": {"gamma": 0.04, "batch_size": 6},
    "DelayAdaptiveMuonASGD": {"gamma": 0.005},
}

# These are the primary knobs that materially affect each method's stability/speed tradeoff.
# Muon's internal parameters are held fixed by default so the comparison focuses on the async method.
TUNING_GRIDS = {
    "RingmasterMuonASGD": {
        "gamma": [5 ** i for i in range(-6, 2)],
        "max_delay": [1, 2, 4, 6, 8, 16, 32],
    },
    "ParameterAgnosticRingmasterMuonASGD": {
        "eta": [5 ** i for i in range(-4, 4)],
    },
    "RennalaMuonSGD": {
        "gamma": [5 ** i for i in range(-6, 2)],
        "batch_size": [1, 2, 4, 6, 8, 16, 32],
    },
    "DelayAdaptiveMuonASGD": {
        "gamma": [5 ** i for i in range(-8, 0)],
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


def build_problem(problem_seed):
    main_diag, side_diag, b = create_worst_case(DIM, 1)
    stochastic_func = StochasticTridiagonalQuadraticFunction(main_diag, side_diag, b, problem_seed, NOISE, "add")
    function = stochastic_func._tridiagonal_quadratic
    analytical_solution = np.linalg.solve(function._A.toarray(), function._b)

    def objective(x):
        return 0.5 * x.T @ function._A @ x - function._b.T @ x

    nodes = [Signature(StochasticGradientNodeAlgorithm, stochastic_func) for _ in range(NUM_NODES)]

    point = np.zeros(DIM)
    point[0] = np.sqrt(DIM)

    return {
        "function": function,
        "objective": objective,
        "solution": analytical_solution,
        "objective_star": objective(analytical_solution),
        "nodes": nodes,
        "point": point,
    }


def build_transport(nodes, delay_seed):
    generator = np.random.default_rng(seed=delay_seed)

    def halfnormal(index):
        return np.abs(generator.normal(loc=0, scale=np.sqrt(index + 1)))

    return RandomDelayedAsynchronousTransport(nodes, DELAYS, halfnormal)


def merge_optimizer_params(method_name, params):
    merged = dict(params)
    if method_name != "ParameterAgnosticRingmasterMuonASGD":
        for key, value in COMMON_MUON_PARAMS.items():
            merged.setdefault(key, value)
    else:
        merged.setdefault("ns_steps", COMMON_MUON_PARAMS["ns_steps"])
        merged.setdefault("nesterov", COMMON_MUON_PARAMS["nesterov"])
    return merged


def build_optimizer(method_name, params, point, transport):
    params = merge_optimizer_params(method_name, params)
    if method_name == "RingmasterMuonASGD":
        return RingmasterMuonASGD(
            transport,
            point.copy(),
            max_delay=params["max_delay"],
            gamma=params["gamma"],
            beta=params["beta"],
            ns_steps=params["ns_steps"],
            nesterov=params["nesterov"],
        )
    if method_name == "ParameterAgnosticRingmasterMuonASGD":
        return ParameterAgnosticRingmasterMuonASGD(
            transport,
            point.copy(),
            eta=params["eta"],
            ns_steps=params["ns_steps"],
            nesterov=params["nesterov"],
        )
    if method_name == "RennalaMuonSGD":
        return RennalaMuonSGD(
            transport,
            point.copy(),
            gamma=params["gamma"],
            batch_size=params["batch_size"],
            beta=params["beta"],
            ns_steps=params["ns_steps"],
            nesterov=params["nesterov"],
        )
    if method_name == "DelayAdaptiveMuonASGD":
        return DelayAdaptiveMuonASGD(
            transport,
            point.copy(),
            gamma=params["gamma"],
            delay_adaptive=True,
            beta=params["beta"],
            ns_steps=params["ns_steps"],
            nesterov=params["nesterov"],
        )
    raise ValueError(f"Unknown method {method_name}")


def run_optimizer(optimizer, function, objective, objective_star, point, time_lim):
    objective_gap = [objective(point) - objective_star]
    gradient_norm_sq = [np.linalg.norm(function.gradient(point)) ** 2]
    runtime = [0.0]

    while runtime[-1] < time_lim:
        optimizer.step()
        current_point = optimizer.get_point()
        objective_gap.append(objective(current_point) - objective_star)
        gradient_norm_sq.append(np.linalg.norm(function.gradient(current_point)) ** 2)
        runtime.append(optimizer.get_time())

    return {
        "runtime": runtime,
        "objective_gap": objective_gap,
        "gradient_norm_sq": gradient_norm_sq,
    }


def evaluate_method(method_name, params, time_lim, problem_seed, delay_seed):
    experiment = build_problem(problem_seed)
    transport = build_transport(experiment["nodes"], delay_seed)
    optimizer = build_optimizer(method_name, params, experiment["point"], transport)
    return run_optimizer(
        optimizer,
        experiment["function"],
        experiment["objective"],
        experiment["objective_star"],
        experiment["point"],
        time_lim,
    )


def score_trace(trace, metric_name):
    return float(trace[metric_name][-1])


def candidate_param_sets(method_name):
    grid = TUNING_GRIDS[method_name]
    keys = list(grid.keys())
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def tune_method(method_name, time_lim, metric_name, num_trials, aggregator):
    best_score = np.inf
    best_params = None
    best_trial_scores = None

    for params in candidate_param_sets(method_name):
        trial_scores = []
        for trial_index in range(num_trials):
            trace = evaluate_method(
                method_name,
                params,
                time_lim,
                problem_seed=PROBLEM_SEED + trial_index,
                delay_seed=TRANSPORT_SEED + trial_index,
            )
            trial_scores.append(score_trace(trace, metric_name))

        aggregate_score = float(np.mean(trial_scores) if aggregator == "mean" else np.median(trial_scores))
        print(
            f"{method_name}: score={aggregate_score:.6e}, "
            f"trial_scores={[float(score) for score in trial_scores]}, params={params}"
        )

        if aggregate_score < best_score:
            best_score = aggregate_score
            best_params = dict(params)
            best_trial_scores = [float(score) for score in trial_scores]

    return {
        "params": best_params,
        "score": float(best_score),
        "metric": metric_name,
        "num_trials": num_trials,
        "aggregator": aggregator,
        "trial_scores": best_trial_scores,
    }


def tune_all_methods(time_lim, metric_name, num_trials, aggregator):
    tuned = {}
    for method_name in PLOTTING_ORDER:
        print(f"Tuning {method_name}...")
        tuned[method_name] = tune_method(method_name, time_lim, metric_name, num_trials, aggregator)
    return tuned


def save_tuned_params(tuned_results, output_path, time_lim, metric_name, num_trials, aggregator):
    payload = {
        "_meta": {
            "time_limit": time_lim,
            "metric": metric_name,
            "num_trials": num_trials,
            "aggregator": aggregator,
            "common_muon_params": COMMON_MUON_PARAMS,
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
        return rf"Ringmaster Muon: $\gamma={params['gamma']}$, $R={params['max_delay']}$"
    if method_name == "ParameterAgnosticRingmasterMuonASGD":
        return rf"PA Ringmaster Muon: $\eta={params['eta']}$"
    if method_name == "RennalaMuonSGD":
        return rf"Rennala Muon: $\gamma={params['gamma']}$, $B={params['batch_size']}$"
    if method_name == "DelayAdaptiveMuonASGD":
        return rf"Delay-Adaptive Muon: $\gamma={params['gamma']}$"
    raise ValueError(f"Unknown method {method_name}")


def metric_to_plot_label(metric_name):
    if metric_name == "objective_gap":
        return r"$f(x^t)-f^{\inf}$"
    if metric_name == "gradient_norm_sq":
        return r"$\|\nabla f(x^t)\|^2$"
    raise ValueError(f"Unknown metric {metric_name}")


def plot_comparison(params_by_method, time_lim, metric_name, title_suffix="", trial_index=0):
    plt.figure()

    for method_name in PLOTTING_ORDER:
        trace = evaluate_method(
            method_name,
            params_by_method[method_name],
            time_lim,
            problem_seed=PROBLEM_SEED + trial_index,
            delay_seed=TRANSPORT_SEED + trial_index,
        )
        print(f"Comparing {method_name} with params={params_by_method[method_name]}")
        plt.semilogy(
            trace["runtime"],
            trace[metric_name],
            label=format_method_label(method_name, params_by_method[method_name]),
            linestyle="solid",
            marker=MARKERS[method_name],
            markersize=MARKER_SIZES[method_name],
            markevery=max(1, len(trace["runtime"]) // 10),
            color=COLORS[method_name],
        )

    plt.legend(loc="upper right", prop={"size": 15})
    plt.xlim(0, time_lim)
    plt.xlabel("Runtime (seconds)")
    plt.ylabel(metric_to_plot_label(metric_name))
    title = "Muon Comparison"
    if title_suffix:
        title = f"{title}: {title_suffix}"
    plt.title(title)
    plt.tight_layout()
    plt.show()


def print_tuning_summary():
    print("Hyperparameters tuned by default:")
    print("  Ringmaster Muon: gamma and max_delay (refresh threshold R).")
    print("  PA Ringmaster Muon: eta only; alpha_k, threshold R_k, and Muon momentum schedule follow the theorem.")
    print("  Rennala Muon: gamma and batch_size B.")
    print("  Delay-Adaptive Muon: gamma only.")
    print("Shared Muon internals kept fixed by default:")
    print(f"  beta={COMMON_MUON_PARAMS['beta']}, ns_steps={COMMON_MUON_PARAMS['ns_steps']}, nesterov={COMMON_MUON_PARAMS['nesterov']}")
    print("These are held fixed so the comparison isolates the async method rather than re-tuning Muon's internals per method.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Tune and compare Muon-based asynchronous methods. "
            "Use --mode tune_and_compare to search hyperparameters first and then plot the methods, "
            "or --mode compare to plot with a given params file."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["tune", "compare", "tune_and_compare"],
        default="compare",
        help="`tune` searches hyperparameters, `compare` plots using a parameter set, `tune_and_compare` does both.",
    )
    parser.add_argument(
        "--params-file",
        default=os.path.join(os.path.dirname(__file__), "ringmaster_muon_tuned_params.json"),
        help="JSON file used to save tuned params to or load params from.",
    )
    parser.add_argument(
        "--time-lim",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help="Runtime horizon used both for tuning and the final comparison plot.",
    )
    parser.add_argument(
        "--tuning-time-lim",
        type=float,
        default=None,
        help="Optional shorter runtime horizon used only during hyperparameter tuning. Defaults to --time-lim.",
    )
    parser.add_argument(
        "--metric",
        choices=["objective_gap", "gradient_norm_sq"],
        default="objective_gap",
        help="Metric optimized during tuning and shown on the comparison plot.",
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
    tuning_time_lim = args.time_lim if args.tuning_time_lim is None else args.tuning_time_lim

    if args.mode in ("tune", "tune_and_compare"):
        tuned_results = tune_all_methods(tuning_time_lim, args.metric, args.num_trials, args.aggregator)
        save_tuned_params(
            tuned_results,
            args.params_file,
            tuning_time_lim,
            args.metric,
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
            args.metric,
            title_suffix="tuned",
            trial_index=args.compare_trial_index,
        )
        return

    if args.use_default_params:
        params_by_method = DEFAULT_PARAMS
        title_suffix = "defaults"
    else:
        params_by_method = load_params(args.params_file)
        title_suffix = os.path.basename(args.params_file)

    plot_comparison(
        params_by_method,
        args.time_lim,
        args.metric,
        title_suffix=title_suffix,
        trial_index=args.compare_trial_index,
    )


if __name__ == "__main__":
    main()
