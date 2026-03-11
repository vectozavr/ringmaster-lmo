# plot_average.py
import os
import glob
import json
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- make nice, editable PDFs ---
mpl.rcParams['pdf.fonttype'] = 42   # TrueType in PDF (editable text)
mpl.rcParams['ps.fonttype']  = 42

METHODS_ORDER = ["ringleader", "malenia", "iaasgd"]

METHOD_MARKER = {
    "ringleader": "*",
    "malenia":    "H",
    "iaasgd":     "^",
}
METHOD_PRETTY = {
    "ringleader": "Ringleader ASGD",
    "malenia": "Malenia SGD",
    "iaasgd": r"$\mathrm{IA^2SGD}$",
}
METHOD_COLOR = {
    "ringleader": "#C00000",
    "malenia":    "#006666",
    "iaasgd":     "#B8860B",
}

# ---------------- smoothing & utility helpers ----------------
def _ensure_odd(n: int) -> int:
    return n if (n % 2 == 1) else n + 1

def moving_avg_log(y, window_pts=101, use_log=True, eps=1e-20, edge_mode="causal_start"):
    """
    Moving average on a uniform index grid, optionally in log-space.
    O(n) via prefix sums.

    edge_mode:
      - "centered": classic centered window [k-half, k+half] clipped to bounds
      - "causal_start": for k < half, use purely causal window [0, k]; otherwise centered
    """
    n = len(y)
    if n == 0:
        return y
    w = _ensure_odd(int(max(1, window_pts)))
    if w > n:
        w = n if (n % 2 == 1) else n - 1
        if w < 1:
            return y
    half = w // 2

    if use_log:
        y_work = np.log(np.maximum(y, eps))
    else:
        y_work = y.astype(float, copy=False)

    ps = np.empty(n + 1, dtype=float); ps[0] = 0.0
    ps[1:] = np.cumsum(y_work)

    out = np.empty_like(y_work)
    for k in range(n):
        if edge_mode == "causal_start" and k < half:
            L, R = 0, k  # purely causal during warm-up
        else:
            L = max(0, k - half)
            R = min(n - 1, k + half)
        s = ps[R + 1] - ps[L]
        out[k] = s / (R - L + 1)

    return np.exp(out) if use_log else out

def smooth_series(y, window_pts, use_log, eps, pin_first=True, floor=None, edge_mode="causal_start"):
    """
    Smooth a single series AFTER aggregation, optionally pin first value
    to the original (unsmoothed) first sample to preserve identical starts.
    Uses causal warm-up near the start to avoid a jump at index 1.
    """
    y = np.asarray(y, dtype=float)
    y0 = float(y[0])
    if floor is not None:
        y = np.maximum(y, floor)
    y_sm = moving_avg_log(y, window_pts=window_pts, use_log=use_log, eps=eps, edge_mode=edge_mode)
    if pin_first:
        y_sm[0] = max(y0, eps if floor is None else floor)
    return y_sm

def ffill_zeros(y, floor):
    """Forward-fill values <= floor with last positive sample, then floor."""
    y = np.asarray(y, dtype=float).copy()
    mask = y <= floor
    if mask.any():
        valid_idx = np.nonzero(~mask)[0]
        last = floor
        if valid_idx.size:
            last = max(y[valid_idx[0]], floor)
        for i in range(y.size):
            if y[i] > floor:
                last = y[i]
            else:
                y[i] = last
    return np.maximum(y, floor)

# ---------------- I/O helpers ----------------
def load_runs(run_dir, metric="grads", dataset_filter=None):
    """
    Load all *.npz, group by method.
    metric: 'grads' or 'loss'
    dataset_filter: None or 'fashion'/'mnist' to filter by meta.dataset.
    """
    paths = glob.glob(os.path.join(run_dir, "*.npz"))
    by_method = {}
    meta_any = None
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            times = z["times"]
            arr   = z[metric]           # 'grads' or 'loss'
            meta  = json.loads(str(z["meta"]))
            if dataset_filter is not None:
                if str(meta.get("dataset", "")).lower() != dataset_filter.lower():
                    continue
            m = meta["method"]
            by_method.setdefault(m, []).append((times, arr))
            meta_any = meta
    return by_method, meta_any

def make_time_grid(by_method, num_points=1001):
    t_max = 0.0
    for runs in by_method.values():
        for (t, _) in runs:
            if t.size > 0:
                t_max = max(t_max, float(t[-1]))
    if t_max <= 0:
        t_max = 1.0
    return np.linspace(0.0, t_max, int(num_points))

def interp_on_grid(t, y, grid):
    y0 = float(y[0]); yN = float(y[-1])
    return np.interp(grid, t, y, left=y0, right=yN)

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="runs_fashion",
                        help="Directory containing saved *.npz runs.")
    parser.add_argument("--num_points", type=int, default=1001,
                        help="Points on the common time grid.")
    parser.add_argument("--outfile", type=str, default="avg_fashion.pdf",
                        help="Output path base; .pdf will be enforced (e.g., avg or avg.pdf).")

    # Metric / filtering
    parser.add_argument("--metric", choices=["grads", "loss"], default="grads",
                        help="Plot 'grads' (||∇f||²) or 'loss'.")
    parser.add_argument("--dataset_filter", type=str, default=None,
                        help="If set, only include runs whose meta.dataset matches (e.g., 'fashion' or 'mnist').")

    # Aggregation choices
    parser.add_argument("--center", choices=["mean", "median", "geom_mean"], default="median",
                        help="Center curve across runs (aggregated BEFORE smoothing).")
    parser.add_argument("--band", choices=["std", "iqr", "p10p90", "none"], default="iqr",
                        help="Uncertainty band type across runs.")

    # Post-aggregation smoothing
    parser.add_argument("--smooth_window_pts", type=int, default=101,
                        help="Window size (points) for moving average applied AFTER aggregation.")
    parser.add_argument("--smooth_in_log", action="store_true",
                        help="Smooth in log-space (recommended with semilogy).")
    parser.add_argument("--no_smooth_in_log", dest="smooth_in_log", action="store_false")
    parser.set_defaults(smooth_in_log=True)

    # Y handling & scale
    parser.add_argument("--y_floor", type=float, default=1e-12,
                        help="Minimum positive floor to avoid zeros on log scales.")
    parser.add_argument("--ffill_zeros", action="store_true",
                        help="Forward-fill values <= y_floor before flooring.")
    parser.add_argument("--scale", choices=["semilogy", "symlog", "log1p", "linear"], default="semilogy",
                        help="Y-axis style.")
    parser.add_argument("--symlog_linthresh", type=float, default=1e-10,
                        help="Linear threshold for symlog scale (around zero).")

    # X limit
    parser.add_argument("--tmax", type=float, default=None,
                        help="If set, cap the x-axis at this time.")

    args = parser.parse_args()

    by_method, meta_any = load_runs(args.run_dir, metric=args.metric, dataset_filter=args.dataset_filter)
    if not by_method:
        raise SystemExit(f"No runs found in {args.run_dir} (metric={args.metric}, dataset_filter={args.dataset_filter})")

    grid = make_time_grid(by_method, num_points=args.num_points)
    fig, ax = plt.subplots(figsize=(10, 7))

    for method in METHODS_ORDER:
        if method not in by_method:
            continue
        runs = by_method[method]

        # --- per-run prep: interpolation + zero handling ONLY (no smoothing here) ---
        per_run = []
        for (t, yraw) in runs:
            y = interp_on_grid(t, yraw, grid)
            if args.ffill_zeros:
                y = ffill_zeros(y, args.y_floor)
            else:
                y = np.maximum(y, args.y_floor)
            per_run.append(y)

        curves = np.stack(per_run, axis=0)  # (num_runs, T)

        # --- aggregate BEFORE smoothing ---
        if args.center == "median":
            center = np.quantile(curves, 0.50, axis=0)
        elif args.center == "geom_mean":
            center = np.exp(np.mean(np.log(np.maximum(curves, args.y_floor)), axis=0))
        else:  # mean
            center = curves.mean(axis=0)

        if args.band == "std":
            s = curves.std(axis=0)
            lower = center - s
            upper = center + s
        elif args.band == "iqr":
            lower = np.quantile(curves, 0.25, axis=0)
            upper = np.quantile(curves, 0.75, axis=0)
        elif args.band == "p10p90":
            lower = np.quantile(curves, 0.10, axis=0)
            upper = np.quantile(curves, 0.90, axis=0)
        else:
            lower = upper = center

        # Enforce floor before smoothing/plotting
        center = np.maximum(center, args.y_floor)
        lower  = np.maximum(lower,  args.y_floor)
        upper  = np.maximum(upper,  args.y_floor)

        # --- POST-AGGREGATION SMOOTHING with causal warm-up + "pin first" ---
        center_unsm_first = float(center[0])
        lower_unsm_first  = float(lower[0])
        upper_unsm_first  = float(upper[0])

        if args.smooth_window_pts and args.smooth_window_pts > 1:
            center = smooth_series(center, args.smooth_window_pts, args.smooth_in_log, args.y_floor,
                                   pin_first=True, floor=args.y_floor, edge_mode="causal_start")
            lower  = smooth_series(lower,  args.smooth_window_pts, args.smooth_in_log, args.y_floor,
                                   pin_first=True, floor=args.y_floor, edge_mode="causal_start")
            upper  = smooth_series(upper,  args.smooth_window_pts, args.smooth_in_log, args.y_floor,
                                   pin_first=True, floor=args.y_floor, edge_mode="causal_start")
            # Ensure the band contains the pinned center at the start:
            lower[0] = min(lower_unsm_first, center[0])
            upper[0] = max(upper_unsm_first, center[0])

        label = METHOD_PRETTY.get(method, method)
        color = METHOD_COLOR.get(method, None)
        marker = METHOD_MARKER.get(method, None)
        markevery = max(1, len(grid) // 12)
        markersize = 12 if marker == "*" else 10

        # --- plot ---
        if args.scale == "semilogy":
            ax.semilogy(
                grid, center, label=label, color=color,
                marker=marker, markersize=markersize, markevery=markevery, linewidth=1.8
            )
            ax.fill_between(grid, lower, upper, alpha=0.2, color=color)
        elif args.scale == "symlog":
            ax.plot(
                grid, center, label=label, color=color,
                marker=marker, markersize=markersize, markevery=markevery, linewidth=1.8
            )
            ax.fill_between(grid, lower, upper, alpha=0.2, color=color)
            ax.set_yscale("symlog", linthresh=args.symlog_linthresh)
        elif args.scale == "log1p":
            eps = args.y_floor
            T = lambda x: np.log10(x + eps)
            ax.plot(
                grid, T(center), label=label, color=color,
                marker=marker, markersize=markersize, markevery=markevery, linewidth=1.8
            )
            ax.fill_between(grid, T(lower), T(upper), alpha=0.2, color=color)
            ylabel = r"$\log_{10}(\|\nabla f(x^t)\|^2 + \epsilon)$" if args.metric == "grads" \
                     else r"$\log_{10}(f(x^t) + \epsilon)$"
            ax.set_ylabel(ylabel)
        else:  # linear
            ax.plot(
                grid, center, label=label, color=color,
                marker=marker, markersize=markersize, markevery=markevery, linewidth=1.8
            )
            ax.fill_between(grid, lower, upper, alpha=0.2, color=color)

    ax.legend(loc='upper right', prop={'size': 18})
    if args.tmax is not None:
        ax.set_xlim(0, float(args.tmax))
    else:
        ax.set_xlim(0, grid[-1])
    # ax.set_xlim(0, 375000)
    ax.set_xlabel("Runtime", fontsize=18)
    ylabel = r"$\|\|\nabla f(x^t)\|\|^2$" if args.metric == "grads" else "Training loss"
    ax.set_ylabel(ylabel, fontsize=18)
    plt.tight_layout()

    # --- always save a PDF (enforce .pdf extension) ---
    out = args.outfile or "avg.pdf"
    root, ext = os.path.splitext(out)
    out_pdf = root + ".pdf" if ext.lower() != ".pdf" else out
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    print(f"Saved PDF: {out_pdf}")

if __name__ == "__main__":
    main()
