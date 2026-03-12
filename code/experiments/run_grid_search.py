import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from asynchronous.algorithm import (
    AsynchronousSGD,
    RingmasterASGD,
    RennalaSGD,
    StochasticGradientNodeAlgorithm,
)
from asynchronous.asynchronous_transport import RandomDelayedAsynchronousTransport
from function import StochasticTridiagonalQuadraticFunction, create_worst_case
from signature import Signature


sns.set(
    style="whitegrid",
    context="talk",
    font_scale=1.2,
    palette=sns.color_palette("bright"),
    color_codes=False,
)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "DejaVu Sans"
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["figure.figsize"] = (10, 7)
matplotlib.rcParams["text.usetex"] = True


def build_problem(dim, num_nodes, seed, noise):
    main_diag, side_diag, b = create_worst_case(dim, 1)
    stochastic_func = StochasticTridiagonalQuadraticFunction(
        main_diag, side_diag, b, seed, noise, "add"
    )
    function = stochastic_func._tridiagonal_quadratic
    analytical_solution = np.linalg.solve(function._A.toarray(), function._b)

    delays = np.array([1 + np.sqrt(i) for i in range(num_nodes)])
    generator = np.random.default_rng(seed=5)

    def halfnormal(index):
        return np.abs(generator.normal(loc=0, scale=np.sqrt(index + 1)))

    nodes = [Signature(StochasticGradientNodeAlgorithm, stochastic_func) for _ in range(num_nodes)]
    transport = RandomDelayedAsynchronousTransport(nodes, delays, halfnormal)

    point = np.zeros(dim)
    point[0] = np.sqrt(dim)

    return {
        "function": function,
        "transport": transport,
        "point": point,
        "analytical_solution": analytical_solution,
    }


def make_optimizer(method_name, transport, point, gamma, sweep_value):
    if method_name == "RingmasterASGD":
        return RingmasterASGD(transport, point, max_delay=sweep_value, gamma=gamma)
    if method_name == "RennalaSGD":
        return RennalaSGD(transport, point, gamma=gamma, batch_size=sweep_value)
    if method_name == "ASGD":
        return AsynchronousSGD(transport, point, gamma=gamma, delay_adaptive=True)
    raise ValueError(f"Unknown optimizer {method_name}")


def run_trace(problem, method_name, gamma, sweep_value, time_lim):
    function = problem["function"]
    transport = problem["transport"]
    point = np.array(problem["point"], copy=True)
    analytical_solution = problem["analytical_solution"]
    objective = lambda x: 0.5 * x.T @ function._A @ x - function._b.T @ x

    optimizer = make_optimizer(method_name, transport, point, gamma, sweep_value)
    iteration_times = [0.0]
    iteration_points = [objective(point) - objective(analytical_solution)]

    while iteration_times[-1] < time_lim:
        optimizer.step()
        current_point = optimizer.get_point()
        iteration_points.append(objective(current_point) - objective(analytical_solution))
        iteration_times.append(optimizer.get_time())

    return np.array(iteration_times), np.array(iteration_points)


def grid_search(problem, method_name, gammas, sweep_values, time_lim, show_plot=True):
    best_performance = np.inf
    best_params = None

    if show_plot:
        plt.figure()

    for gamma in gammas:
        current_sweep_values = sweep_values if method_name != "ASGD" else [None]

        for sweep_value in current_sweep_values:
            times, values = run_trace(problem, method_name, gamma, sweep_value, time_lim)
            performance = values[-1]

            label = rf"$\gamma={gamma}$"
            if method_name == "RingmasterASGD":
                label += rf", $R={sweep_value}$"
            elif method_name == "RennalaSGD":
                label += rf", $B={sweep_value}$"

            if show_plot:
                plt.semilogy(times, values, label=label, alpha=0.8)

            print(
                "method:",
                method_name,
                "performance:",
                performance,
                "gamma:",
                gamma,
                "sweep_value:",
                sweep_value,
            )
            if performance < best_performance:
                best_performance = performance
                best_params = (gamma, sweep_value)

    if show_plot:
        plt.xlim(0, time_lim)
        plt.xlabel("Runtime (seconds)")
        plt.ylabel(r"$f(x^t)-f^{\inf}$")
        plt.title(f"{method_name} grid search")
        plt.legend(loc="best", fontsize=10)
        plt.tight_layout()
        plt.show()

    return best_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["RingmasterASGD"],
        choices=["RingmasterASGD", "RennalaSGD", "ASGD"],
        help="Methods to grid-search.",
    )
    parser.add_argument("--dim", type=int, default=1729)
    parser.add_argument("--num_nodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--time_lim", type=float, default=5 * 1e3)
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[5**i for i in range(-5, -1)],
        help="Gamma values to sweep.",
    )
    parser.add_argument(
        "--sweep_values",
        type=int,
        nargs="+",
        default=[100, 25, 6],
        help="Delay or batch-size sweep values. Ignored for ASGD.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable plotting and print only the best parameters.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    problem = build_problem(args.dim, args.num_nodes, args.seed, args.noise)

    for method_name in args.methods:
        best_gamma, best_sweep_value = grid_search(
            problem=problem,
            method_name=method_name,
            gammas=args.gammas,
            sweep_values=args.sweep_values,
            time_lim=args.time_lim,
            show_plot=not args.no_plot,
        )

        if method_name == "RingmasterASGD":
            print(f"Best {method_name}: gamma={best_gamma}, max_delay={best_sweep_value}")
        elif method_name == "RennalaSGD":
            print(f"Best {method_name}: gamma={best_gamma}, batch_size={best_sweep_value}")
        else:
            print(f"Best {method_name}: gamma={best_gamma}")


if __name__ == "__main__":
    main()
