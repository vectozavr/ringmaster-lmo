import numpy as np

def reduce_dataset_to_multiple(features, labels, number_of_classes, num_nodes, seed=42):
    """
    Trim dataset so len(labels) % num_nodes == 0 by removing as few samples as possible.
    Ensures trimming is proportional across classes and correctness is verified.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    N = len(labels)
    C = int(number_of_classes)

    # --- Safety guards ---
    assert num_nodes > 0, "num_nodes must be positive"
    assert N >= num_nodes, f"Dataset has only {N} samples, fewer than num_nodes={num_nodes}"
    assert np.all(labels >= 0), "Labels must be non-negative integers"
    assert np.all(labels < C), "Labels must be < number_of_classes"

    # target size (largest multiple of num_nodes ≤ N)
    N_target = (N // num_nodes) * num_nodes
    r = N - N_target

    # If already divisible, return unchanged
    if r == 0:
        kept_idx = np.arange(N)
        dropped_idx = np.array([], dtype=int)

        # Check invariants
        assert len(kept_idx) == N
        assert len(dropped_idx) == 0
        assert (len(kept_idx) % num_nodes) == 0
        return features, labels, kept_idx, dropped_idx

    # class-wise counts
    idx_by_class = [np.where(labels == c)[0] for c in range(C)]
    counts = np.array([len(ix) for ix in idx_by_class], dtype=int)

    # Proportional allocation of drops
    p = counts / counts.sum()
    raw_drop = p * r
    base_drop = np.floor(raw_drop).astype(int)
    short = r - base_drop.sum()
    if short > 0:
        frac_order = np.argsort(-(raw_drop - base_drop))
        base_drop[frac_order[:short]] += 1

    assert base_drop.sum() == r, "Drop allocation mismatch"

    # Randomly select indices to drop
    drop_list = []
    for c in range(C):
        k = base_drop[c]
        if k > 0:
            assert len(idx_by_class[c]) >= k, "Not enough samples in class to drop"
            drop_list.append(rng.choice(idx_by_class[c], size=k, replace=False))
    dropped_idx = np.concatenate(drop_list) if drop_list else np.array([], dtype=int)

    # Build kept set
    mask = np.ones(N, dtype=bool)
    mask[dropped_idx] = False
    kept_idx = np.nonzero(mask)[0]

    # --- Assertions for correctness ---
    assert len(kept_idx) + len(dropped_idx) == N, "Lost or duplicated samples"
    assert len(set(kept_idx) & set(dropped_idx)) == 0, "Overlap between kept and dropped"
    assert (len(kept_idx) % num_nodes) == 0, "Kept size not divisible by num_nodes"
    union = np.sort(np.concatenate([kept_idx, dropped_idx]))
    assert union.shape == (N,) and np.array_equal(union, np.arange(N)), \
        "Kept + dropped indices do not cover the full dataset"

    # Optional: check class balance approx
    before_counts = np.bincount(labels, minlength=C)
    after_counts = np.bincount(labels[kept_idx], minlength=C)
    # Difference should sum to r
    assert before_counts.sum() - after_counts.sum() == r

    return features[kept_idx], labels[kept_idx], kept_idx, dropped_idx


def dirichlet_partition_equal_size(
    features, labels, number_of_classes,
    num_nodes=10, alpha=0.5, seed=0
):
    """
    Equal-size Dirichlet partitioner that uses ALL of the (already trimmed) data.
    Assumes len(labels) % num_nodes == 0.
    Returns client-wise features/labels and the index splits (relative to the *trimmed* arrays).
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    N = len(labels)
    C = int(number_of_classes)
    K = int(num_nodes)
    assert N % K == 0, "Dataset size must be divisible by num_nodes (trim first)."

    N_per_client = N // K

    # per-class pools (over trimmed data)
    idx_by_class = [np.where(labels == c)[0].tolist() for c in range(C)]
    for pool in idx_by_class:
        rng.shuffle(pool)

    def rounded_counts(p, total):
        raw = p * total
        base = np.floor(raw).astype(int)
        short = total - base.sum()
        if short > 0:
            frac_order = np.argsort(-(raw - base))
            base[frac_order[:short]] += 1
        return base

    idx_splits = [[] for _ in range(K)]
    for j in range(K):
        # Dirichlet proportions for this client
        p = rng.dirichlet(np.ones(C) * alpha)
        need_c = rounded_counts(p, N_per_client)

        # Extract; top-up if a class runs short to keep exact size
        for c in range(C):
            need = int(need_c[c])
            take = min(need, len(idx_by_class[c]))
            if take > 0:
                idx_splits[j].extend(idx_by_class[c][:take])
                idx_by_class[c] = idx_by_class[c][take:]
            deficit = need - take
            if deficit > 0:
                # top-up from classes with remaining stock
                remaining = np.array([len(pool) for pool in idx_by_class])
                for c2 in np.argsort(-remaining):
                    if deficit == 0 or remaining[c2] == 0:
                        continue
                    give = min(deficit, len(idx_by_class[c2]))
                    idx_splits[j].extend(idx_by_class[c2][:give])
                    idx_by_class[c2] = idx_by_class[c2][give:]
                    deficit -= give
                    if deficit == 0:
                        break

        rng.shuffle(idx_splits[j])

    # Materialize client datasets
    client_features = [features[idxs] for idxs in idx_splits]
    client_labels   = [labels[idxs]  for idxs in idx_splits]

    # safety checks: union == trimmed data, no dups, equal sizes
    flat = np.concatenate(idx_splits)
    assert len(flat) == N and len(np.unique(flat)) == N
    assert all(len(ix)==N_per_client for ix in idx_splits)

    return client_features, client_labels, idx_splits
