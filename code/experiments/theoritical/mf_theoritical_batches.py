import numpy as np

def MindFlayerBatchSize(ps, clipping_times, eps, sigma2):
    num_nodes = len(clipping_times)

    taus_plus_ts = np.array(clipping_times)
    index_mapping = np.argsort(taus_plus_ts)
    original_order = np.empty_like(index_mapping)
    original_order[index_mapping] = np.arange(len(index_mapping))

    taus_plus_ts_sorted = taus_plus_ts[index_mapping]
    ps_sorted = np.array(ps)[index_mapping]

    S = max(1.0, sigma2 / eps)

    t_m_values = []
    for m in range(num_nodes):
        sum_pj = np.sum(ps_sorted[:m+1])
        sum_pj_over_tau_t = np.sum([ps[j]/taus_plus_ts_sorted[j] for j in range(m+1)])
        t_m = (sum_pj_over_tau_t ** -1) * (S + sum_pj)
        t_m_values.append(t_m)

    m_star = np.argmin(t_m_values)
    t_m_star = min(t_m_values)

    B = np.zeros(num_nodes)
    for i in range(num_nodes):
        if i <= m_star:
            B[i] = np.ceil(t_m_star / taus_plus_ts_sorted[i] - 1)
    return B[original_order]
