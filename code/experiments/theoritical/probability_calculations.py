import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from joblib import Parallel, delayed
from tqdm.auto import tqdm

def cdf(x, **kwargs):
    delay_type = kwargs.get("type")
    if delay_type == "exponential":
        return stats.gamma.cdf(x, a=1, scale=kwargs.get("scale"))
    elif delay_type == "inf_bernoulli":
        return 1 if x < 0 else kwargs.get("p")
    elif delay_type == "alpha_bernoulli":
        scaled_x = x / ((kwargs.get("alpha") - 1) * kwargs.get("tau_i"))
        return stats.bernoulli.cdf(scaled_x, 1 - kwargs.get("p"))
    elif delay_type == "levy":
        return stats.levy.cdf(x, scale=kwargs.get("scale"),
                                  loc=kwargs.get("loc"))
    elif delay_type == "cauchy":
        loc = kwargs.get("loc")
        scale = kwargs.get("scale")
        return stats.cauchy.cdf((x-loc)/scale) -\
                stats.cauchy.cdf((-x-loc)/scale)
    elif delay_type == "logcauchy":
        return stats.cauchy.cdf(np.log(x), scale=kwargs.get("scale"),
                                  loc=kwargs.get("loc"))
    elif delay_type == "levy_exp_bernoulli":
        scale1 = kwargs.get("scale1")
        scale2 = kwargs.get("scale2")
        loc2 = kwargs.get("loc2")
        prob = kwargs.get("p")
        if x < 0:
            return 0
        elif 0 <= x <= kwargs.get("loc2"):
            return prob * stats.gamma.cdf(x, a=1, scale=scale1)
        else:
            return prob * stats.gamma.cdf(x, a=1, scale=scale1) +\
                    (1 - prob) * stats.levy.cdf(x, scale=scale2, loc=loc2)
    elif delay_type == "lognorm":
        return stats.lognorm.cdf(x, s=kwargs.get('s'))
    elif delay_type == "heterolognorm":
        return stats.lognorm.cdf(x, s=kwargs.get('s') * kwargs.get("het_func")(kwargs.get('i')))
    else:
        raise ValueError(f"Unsupported delay_type '{delay_type}")


def sum_cdf(x, num_summands, **kwargs):
    delay_type = kwargs.get("type")
    if delay_type == "exponential":
        return stats.gamma.cdf(x, a=num_summands, scale=kwargs.get("scale"))
    elif delay_type == "inf_bernoulli":
        p_zero_all = 1
        for _ in range(num_summands):
            p_zero_all *= kwargs.get("p")
        return 1 if x < 0 else p_zero_all
    elif delay_type == "alpha_bernoulli":
        scaled_x = x / ((kwargs.get("alpha") - 1) * kwargs.get("tau_i"))
        return stats.binom.cdf(scaled_x, num_summands, 1 - kwargs.get("p"))
    elif delay_type == "levy":
        return stats.levy.cdf(x, scale=num_summands * kwargs.get("scale"),
                              loc=num_summands * kwargs.get("loc"))
    # elif delay_type == "cauchy":
    # Not clear how to implement the sum of abs cauchy
    # elif delay_type == "levy_exp_bernoulli":
    # also not clear how to implement, but maybe this is easier?
    else:
        raise ValueError(f"Unsupported delay_type '{delay_type}")


# General purpose that should be used in the gradient estimator
def calculate_p0_tilde(T, **kwargs):
    delay = kwargs.get("func")
    num_nodes = kwargs.get("num_nodes")

    p_0 = 0
    for i in range(num_nodes):
        kwargs["tau_i"] = delay(i)
        p_0 += 1 - cdf(T - delay(i), **kwargs)
    p_0 /= num_nodes

    return p_0


def generate_combinations_with_bounds(bounds, target_sum):
    def find_combinations(bounds, target, current_combination, current_sum, start):
        if current_sum == target:
            completed_combination = current_combination + [0] * (len(bounds) - len(current_combination))
            yield completed_combination
            return
        if start >= len(bounds):
            return

        upper_bound = bounds[start]
        for value in range(0, upper_bound + 1):
            if current_sum + value <= target:
                yield from find_combinations(bounds, target, current_combination + [value], current_sum + value, start + 1)

    return find_combinations(bounds, target_sum, [], 0, 0)


def calculate_pB(T, m, **kwargs):
    pB = 0
    delay = kwargs.get("func")
    n = kwargs.get("num_nodes")

    for combination in generate_combinations_with_bounds([int(np.floor(T / delay(i))) for i in range(n)], m):
        prod = 1
        for i, ki in enumerate(combination):
            prod *= calculate_pimi(T, i, ki, **kwargs)
        pB += prod
    return pB


def mf_calculate_ps(clipping_times, **kwargs):
    delay = kwargs.get("func")
    ps = []
    for i, ct in enumerate(clipping_times):
        kwargs["i"] = i
        ps.append(cdf(ct - delay(i), **kwargs))
    return ps

def modmf_calculate_clipping_times(p, **kwargs):
    # for now this is fine
    assert kwargs.get("type") == "heterolognorm"
    return [stats.lognorm.ppf(p, s=kwargs.get('s') * kwargs.get("het_func")(i)) for i in\
            range(kwargs.get("num_nodes"))]

# General purpose that should be used in the gradient estimator
def calculate_p0(T, **kwargs):
    delay = kwargs.get("func")
    num_nodes = kwargs.get("num_nodes")

    p_0 = 1
    if kwargs.get("is_independent"):
        for i in range(num_nodes):
            kwargs["tau_i"] = delay(i)
            p_0 *= 1 - cdf(T - delay(i), **kwargs)
    else:
        tau_min = min([delay(i) for i in range(num_nodes)])
        kwargs["tau_i"] = tau_min
        p_0 *= 1 - cdf(T - tau_min, **kwargs)

    return p_0


def calculate_pimi(T, idx, mi, **kwargs):
    delay = kwargs.get("func")
    kwargs["tau_i"] = delay(idx)

    if mi == 0:
        return 1 - cdf(T - delay(idx), **kwargs)

    term1 = sum_cdf(T - mi*delay(idx), mi, **kwargs)
    term2 = sum_cdf(T - (mi+1)*delay(idx), mi+1, **kwargs)
    return term1 - term2


def collection_est_conv(T, sigma2_over_eps, bound="max", **kwargs):
    delay = kwargs.get("func")
    n = kwargs.get("num_nodes")

    p0 = calculate_p0(T, **kwargs)
    max_m = int(sum(np.floor(T/delay(i)) for i in range(n)))
    sum_factor = sum(1/m * calculate_pB(T, m, **kwargs) for m in range(1, max_m+1))

    if bound == "sum":
        return T * (1 + 2 * sigma2_over_eps * sum_factor / (1-p0) ** 2)
    elif bound == "max":
        return T * max(1, 2 * sigma2_over_eps * sum_factor / (1-p0) ** 2)


def client_est_conv(T, sigma2_over_eps, bound="max", **kwargs):
    n = kwargs.get("num_nodes")
    delay = kwargs.get("func")

    p0 = calculate_p0_tilde(T, **kwargs)

    sum_factor = 0
    for i in range(n):
        max_b = int(np.floor(T / delay(i)))
        for b in range(1, max_b + 1):
            sum_factor += calculate_pimi(T, i, b, **kwargs)

    if bound == "sum":
        return T * (1 + 2 * sigma2_over_eps * sum_factor / (n ** 2 * (1 - p0) ** 2))
    if bound == "max":
        return T * max(1, 2 * sigma2_over_eps * sum_factor / (n ** 2 * (1 - p0) ** 2))


if __name__ == "__main__":
    sigma2_over_eps = 5
    n = 100

    random_time_setup = {
        "type": "levy",
        "func": lambda x: np.sqrt(x+1),
        "scale": 1/2,
        "loc": 0,
        "num_nodes": n
    }

    delays = [random_time_setup["func"](i) for i in range(n)]

    # Generate T_values
    T_values = np.linspace(min(delays)+0.01, 30*min(delays), 3000)

    # Parallel computation of function values
    func_values = Parallel(n_jobs=-1)(delayed(client_est_conv)(T, sigma2_over_eps, **random_time_setup) for T in tqdm(T_values))

    # Find the minimum value and its corresponding T
    min_value = min(func_values)
    min_index = func_values.index(min_value)
    min_T = T_values[min_index]

    # Plotting
    plt.semilogy(T_values, func_values)
    plt.scatter([min_T], [min_value], color='red', label=f'Minimum at T={min_T:.2f}')
    plt.xlabel('Time Frame T')
    plt.ylabel('Time of Convergence')
    plt.title('Trend of Convergence as a Function of T')
    plt.grid(True)
    plt.legend()
    plt.show()
