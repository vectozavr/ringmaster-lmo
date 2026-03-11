import numpy as np

def RennalaStepSize(stochastic_func, batch_size, eps, sigma2):
    S = batch_size

    L = np.max(np.linalg.eigvals(stochastic_func._tridiagonal_quadratic._A.toarray()))

    return min(1/L, eps * S / (2 * L * sigma2))


def MindFlayerStepSize(stochastic_func, ps, num_clips, eps, sigma2):
    # Assuming all p_i's are the same which makes sense in the case where we
    # have the same distribution
    B = num_clips.dot(ps)

    L = np.max(np.linalg.eigvals(stochastic_func._tridiagonal_quadratic._A.toarray()))

    return min(1/(2 * L), eps * B / (2 * L * sigma2))


def ModMindFlayerStepSize(stochastic_func, batch_size, p, eps, sigma2):
    B = batch_size * p
    L = np.max(np.linalg.eigvals(stochastic_func._tridiagonal_quadratic._A.toarray()))

    return min(1/(2 * L), eps * B / (2 * L * sigma2))


def ASGDStepSize(stochastic_func, eps, sigma2):
    L = np.max(np.linalg.eigvals(stochastic_func._tridiagonal_quadratic._A.toarray()))

    return 1/(2 * L)
