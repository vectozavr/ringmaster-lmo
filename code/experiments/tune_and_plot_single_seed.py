# tune_and_plot_single_seed.py
import os
import argparse
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

from asynchronous.asynchronous_transport import RandomDelayedAsynchronousTransport
from asynchronous.algorithm import StochasticGradientNodeAlgorithm, RingleaderASGD, MaleniaSGD, IAASGD
from signature import Signature
from utils_NN import prepare_dataset  # now supports dataset={'mnist','fashion'}
from model import SimpleNeuralNetFunction
from prep_data import dirichlet_partition_equal_size, reduce_dataset_to_multiple

METHODS = ["ringleader", "malenia", "iaasgd"]
METHOD_PRETTY = {
    "ringleader": "Ringleader ASGD",
    "malenia":    "Malenia SGD",
    "iaasgd":     r"$\mathrm{IA^2SGD}$",
}
METHOD_COLOR = {
    "ringleader": "#C00000",
    "malenia":    "#006666",
    "iaasgd":     "#B8860B",
}
METHOD_MARKER = {
    "ringleader": "*",
    "malenia":    "H",
    "iaasgd":     "^",
}

Y_FLOOR = 1e-12  # avoid log(0) in semilogy

# ---------- helpers: saving & loading ----------
def _fmt_gamma(g):
    s = f"{g:g}"
    if "e" in s or "E" in s:
        return s.replace("E", "e").replace("+", "")
    return s.replace(".", "p")

def runs_dir(output_dir, num_nodes):
    """Per-num_nodes runs folder, e.g., <output_dir>/runs/N100/"""
    d = os.path.join(output_dir, "runs", f"N{int(num_nodes)}")
    os.makedirs(d, exist_ok=True)
    return d

def run_filepath(output_dir, method, seed, gamma, num_nodes):
    # num_nodes is encoded by the folder; filename kept concise
    fname = f"{method}_gamma{_fmt_gamma(gamma)}_seed{int(seed)}.npz"
    return os.path.join(runs_dir(output_dir, num_nodes), fname)

def save_run(output_dir, method, seed, gamma, num_nodes, time_lim, alpha, subset,
             times, grads, losses, extra_meta=None):
    path = run_filepath(output_dir, method, seed, gamma, num_nodes)
    meta = {
        "method": method,
        "gamma": float(gamma),
        "seed": int(seed),
        "num_nodes": int(num_nodes),
        "time_lim": float(time_lim),
        "alpha": float(alpha) if alpha is not None else None,
        "subset": int(subset) if subset is not None else None,
    }
    if extra_meta:
        meta.update(extra_meta)
    np.savez_compressed(path,
                        times=np.asarray(times),
                        grads=np.asarray(grads),
                        loss=np.asarray(losses),
                        meta=json.dumps(meta))
    return path

def load_run(output_dir, method, seed, gamma, num_nodes):
    path = run_filepath(output_dir, method, seed, gamma, num_nodes)
    with np.load(path, allow_pickle=True) as z:
        times = z["times"]
        grads = z["grads"]
        losses = z["loss"] if "loss" in z.files else None
        meta = json.loads(str(z["meta"]))
    return times, grads, losses, meta, path

# ---------- factories (isolated per run) ----------
def make_delays(num_nodes: int, seed: int) -> np.ndarray:
    base = np.array([1 + i for i in range(num_nodes)], dtype=int)
    rng = np.random.default_rng(seed=seed)
    return rng.permutation(base)

def make_noise_function(num_nodes: int, seed: int):
    gens = [np.random.default_rng(seed + i) for i in range(num_nodes)]
    def halfnormal(index):
        return abs(gens[index].normal(loc=0.0, scale=np.sqrt(index + 1)))
    return halfnormal

def make_transport(client_X, client_y, delays, seed: int):
    nodes = [Signature(StochasticGradientNodeAlgorithm,
                       SimpleNeuralNetFunction(X, y, seed=seed))
             for (X, y) in zip(client_X, client_y)]
    noise_fn = make_noise_function(num_nodes=len(nodes), seed=seed)
    return RandomDelayedAsynchronousTransport(nodes, delays.copy(), noise_fn)

def run_one_method(method_name: str, gamma: float, features, labels,
                   client_X, client_y, delays, time_lim: float, seed: int):
    function  = SimpleNeuralNetFunction(features, labels, seed=seed)
    init_point = np.array(function.get_current_point(), copy=True)
    transport = make_transport(client_X, client_y, delays, seed)

    if method_name == "ringleader":
        Opt = RingleaderASGD
    elif method_name == "malenia":
        Opt = MaleniaSGD
    elif method_name == "iaasgd":
        Opt = IAASGD
    else:
        raise ValueError(f"Unknown method {method_name}")

    optimizer = Opt(transport, init_point, gamma=gamma)

    times  = [0.0]
    grads  = [np.linalg.norm(function.gradient(init_point))**2]
    losses = [float(function.value(init_point))]

    while times[-1] < time_lim:
        optimizer.step()
        times.append(optimizer.get_time())
        pt = optimizer.get_point()
        grads.append(np.linalg.norm(function.gradient(pt))**2)
        losses.append(float(function.value(pt)))

    return np.array(times), np.array(grads), np.array(losses)

def load_or_run_then_save(output_dir, method, gamma, seed, num_nodes, time_lim, alpha, subset,
                          features, labels, client_X, client_y):
    """Load saved run if present; otherwise run, save, and return arrays (including loss)."""
    path = run_filepath(output_dir, method, seed, gamma, num_nodes)
    if os.path.exists(path):
        t, g, l, meta, _ = load_run(output_dir, method, seed, gamma, num_nodes)
        if l is not None:
            return t, g, l, meta, path
        # Backfill: existing file lacks loss -> recompute & overwrite once
    delays = make_delays(num_nodes, seed)
    t, g, l = run_one_method(method, gamma, features, labels, client_X, client_y, delays, time_lim, seed)
    save_run(output_dir, method, seed, gamma, num_nodes, time_lim, alpha, subset, t, g, l,
             extra_meta={"phase": "tuning_or_plot"})
    return t, g, l, {"method": method, "gamma": float(gamma), "seed": int(seed),
                     "num_nodes": int(num_nodes)}, path

# ---------- metrics on Grads ----------
def metric_final(times: np.ndarray, grads: np.ndarray) -> float:
    return float(grads[-1])

def metric_auc(times: np.ndarray, grads: np.ndarray) -> float:
    return float(np.trapz(grads, times))

# ---------- parse method->seed map like "ringleader=26,malenia=27,iaasgd=28" ----------
def parse_method_seed_map(s: str):
    mapping = {}
    if not s:
        return mapping
    for part in s.split(","):
        k, v = part.split("=", 1)
        k = k.strip().lower()
        mapping[k] = int(v.strip())
    return mapping

# ---------- main ----------
def parse_args():
    p = argparse.ArgumentParser(description="Tune stepsize (gamma) for a SINGLE seed (by LOSS); save runs; plot by reading saved runs.")
    p.add_argument("--output_dir", type=str, default="tuning_single_seed")
    p.add_argument("--seed", type=int, default=26)
    p.add_argument("--num_nodes", type=int, default=100)  # clients
    p.add_argument("--time_lim", type=float, default=300000)
    p.add_argument("--alpha", type=float, default=0.1)    # Dirichlet concentration
    p.add_argument("--dataset_root", type=str, default="./")
    p.add_argument("--subset", type=int, default=None, help="Use only the first N examples (before partitioning)")
    p.add_argument("--dataset", type=str, choices=["mnist", "fashion"], default="fashion")

    p.add_argument("--grid_ringleader", type=float, nargs="+",
                   default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--grid_iaasgd", type=float, nargs="+",
                   default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--grid_malenia", type=float, nargs="+",
                   default=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0])

    p.add_argument("--select_metric", choices=["final", "auc"], default="final")  # LOSS-based selection
    p.add_argument("--method_seed_map", type=str, default="")
    p.add_argument("--plot_tuned", type=str, default="tuned_single_seed.png")
    p.add_argument("--plot_diff_seeds", type=str, default="methods_different_seeds.png")
    p.add_argument("--per_method_prefix", type=str, default="gamma_sweep_")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Ensure per-N folder exists
    runs_dir(args.output_dir, args.num_nodes)

    # ---------- dataset selection (uses unified prepare_dataset) ----------
    # prepare_dataset(path_to_dataset, number_of_first_examples=None, dataset='mnist'|'fashion')
    features, labels, number_of_classes = prepare_dataset(
        args.dataset_root,
        number_of_first_examples=args.subset,
        dataset=args.dataset
    )

    # Make length divisible by num_nodes; then Dirichlet partition
    features, labels, kept_idx, dropped_idx = reduce_dataset_to_multiple(
        features, labels, number_of_classes, args.num_nodes, seed=123456
    )
    client_X, client_y, client_idxs = dirichlet_partition_equal_size(
        features, labels, number_of_classes,
        num_nodes=args.num_nodes, alpha=args.alpha, seed=654321
    )

    delays_tune = make_delays(args.num_nodes, args.seed)

    method_grids = {
        "ringleader": args.grid_ringleader,
        "malenia":    args.grid_malenia,
        "iaasgd":     args.grid_iaasgd,
    }

    # ---- tune on single seed (by LOSS) ----
    records = []
    best_gamma = {}

    for method in METHODS:
        grid = method_grids[method]
        best_val = float("inf")
        best_g = None

        for gamma in grid:
            path = run_filepath(args.output_dir, method, args.seed, gamma, args.num_nodes)
            if os.path.exists(path):
                t, g, l, meta, _ = load_run(args.output_dir, method, args.seed, gamma, args.num_nodes)
                if l is None:
                    # backfill loss by recomputing once
                    t, g, l = run_one_method(method, gamma, features, labels, client_X, client_y,
                                             delays_tune, time_lim=args.time_lim, seed=args.seed)
                    save_run(args.output_dir, method, args.seed, gamma, args.num_nodes,
                             args.time_lim, args.alpha, args.subset, t, g, l,
                             extra_meta={"phase": "tuning"})
            else:
                # compute with tuning seed & save
                t, g, l = run_one_method(method, gamma, features, labels, client_X, client_y,
                                         delays_tune, time_lim=args.time_lim, seed=args.seed)
                save_run(args.output_dir, method, args.seed, gamma, args.num_nodes,
                         args.time_lim, args.alpha, args.subset, t, g, l,
                         extra_meta={"phase": "tuning"})

            final = metric_final(t, g)
            auc   = metric_auc(t, g)
            records.append({"method": method, "gamma": gamma, "seed": args.seed,
                            "final_loss": final, "auc_loss": auc})
            score = final if args.select_metric == "final" else auc
            if score < best_val:
                best_val = score
                best_g = gamma

        best_gamma[method] = float(best_g)
        print(f"[tune] {method:10s} best γ={best_g} by {args.select_metric} (LOSS score {best_val:.3e})")

    # save tuning table/summary
    table_path = os.path.join(args.output_dir, "tuning_table_single_seed.csv")
    with open(table_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method","gamma","seed","final_loss","auc_loss"])
        w.writeheader(); w.writerows(records)
    with open(os.path.join(args.output_dir, "tuning_summary_single_seed.json"), "w") as f:
        json.dump({"config": {
                        "seed": args.seed, "num_nodes": args.num_nodes,
                        "time_lim": args.time_lim, "alpha": args.alpha,
                        "subset": args.subset, "select_metric": args.select_metric,
                        "dataset": args.dataset},
                   "best_gamma": best_gamma}, f, indent=2)

    # ---- Plot A: tuned curves (same seed for all) — LOSS ----
    plt.figure(figsize=(10,7))
    for m in METHODS:
        gamma = best_gamma[m]
        t, g, l, meta, path = load_run(args.output_dir, m, args.seed, gamma, args.num_nodes)
        if l is None:
            # ensure we have loss (recompute once)
            t, g, l = run_one_method(m, gamma, features, labels, client_X, client_y,
                                     delays_tune, time_lim=args.time_lim, seed=args.seed)
            save_run(args.output_dir, m, args.seed, gamma, args.num_nodes,
                     args.time_lim, args.alpha, args.subset, t, g, l,
                     extra_meta={"phase": "plot_tuned"})
        y = np.maximum(g, Y_FLOOR)
        plt.semilogy(t, y, label=f"{METHOD_PRETTY[m]}: $\\gamma={gamma}$",
                     color=METHOD_COLOR[m], marker=METHOD_MARKER[m],
                     markevery=max(1, len(t)//10), linewidth=1.8)
    plt.legend(loc='upper right', prop={'size': 14})
    plt.xlabel("Runtime (seconds)")
    plt.ylabel(r"$\|\nabla f(x)\|^2$")
    plt.xlim(0, args.time_lim); plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, args.plot_tuned), dpi=200)

    # ---- Plot B/C/D — per-method gamma sweeps (LOSS), distinct style per γ ----
    linestyles = ["-", "--", "-.", ":"]
    markers_all = ["o", "s", "^", "v", "D", "P", "X", "*", "H", ">", "<"]

    for m in METHODS:
        pretty = METHOD_PRETTY[m]
        grid = sorted(method_grids[m])           # consistent legend
        best_g = best_gamma[m]

        cmap_name = "tab10" if len(grid) <= 10 else "tab20"
        cmap = plt.get_cmap(cmap_name)

        plt.figure(figsize=(10,7))
        for j, gamma in enumerate(grid):
            # ensure run exists with loss, else compute & save
            t, g, l, meta, path = load_or_run_then_save(args.output_dir, m, gamma, args.seed,
                                                        args.num_nodes, args.time_lim, args.alpha, args.subset,
                                                        features, labels, client_X, client_y)

            color = cmap(j % cmap.N)
            ls = linestyles[j % len(linestyles)]
            mk = markers_all[j % len(markers_all)]

            is_best = np.isclose(gamma, best_g)
            lw = 2.6 if is_best else 1.6
            alpha = 1.0 if is_best else 0.9
            label = rf"γ={gamma:g}" + ("  (best)" if is_best else "")

            y = np.maximum(g, Y_FLOOR)
            plt.semilogy(
                t, y,
                label=label,
                color=color,
                linestyle=ls,
                marker=mk,
                markersize=6,
                markevery=max(1, len(t)//12),
                linewidth=lw,
                alpha=alpha,
            )

        plt.title(f"{pretty} — stepsize sweep (seed={args.seed})")
        plt.xlabel("Runtime (seconds)")
        plt.ylabel(r"$\|\nabla f(x)\|^2$")
        plt.xlim(0, args.time_lim)
        ncols = 2 if len(grid) <= 8 else 3
        plt.legend(title="stepsize γ", loc="upper right", prop={"size": 11}, ncol=ncols, frameon=True)
        plt.tight_layout()
        out = os.path.join(args.output_dir, f"{args.per_method_prefix}{m}.png")
        plt.savefig(out, dpi=200)

    # ---- Optional: each method with its own seed (keeps tuned γ) — LOSS ----
    if args.method_seed_map:
        mapping = parse_method_seed_map(args.method_seed_map)
        plt.figure(figsize=(10,7))
        for m in METHODS:
            gamma = best_gamma[m]
            m_seed = mapping.get(m, args.seed)
            delays_m = make_delays(args.num_nodes, m_seed)
            # recompute/ensure run with that seed
            t, g, l = run_one_method(m, gamma, features, labels, client_X, client_y,
                                     delays_m, time_lim=args.time_lim, seed=m_seed)
            save_run(args.output_dir, m, m_seed, gamma, args.num_nodes,
                     args.time_lim, args.alpha, args.subset, t, g, l,
                     extra_meta={"phase": "diff_seeds"})
            y = np.maximum(l, Y_FLOOR)
            plt.semilogy(t, y, label=f"{METHOD_PRETTY[m]} (seed={m_seed}), $\\gamma={gamma}$",
                         color=METHOD_COLOR[m], marker=METHOD_MARKER[m],
                         markevery=max(1, len(t)//10), linewidth=1.8)
        plt.legend(loc='upper right', prop={'size': 14})
        plt.xlabel("Runtime (seconds)"); plt.ylabel("Training loss")
        plt.xlim(0, args.time_lim); plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, args.plot_diff_seeds), dpi=200)

if __name__ == "__main__":
    main()
