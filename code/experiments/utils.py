import numpy as np
import scipy.stats as sps
from itertools import product
import pandas as pd


from function import create_worst_case
from asynchronous.algorithm import AsynchronousSGD,\
        RennalaSGD, MiniBatchSGD, AsynchronousTimeFrameMiniBatchSGD, MindFlayerClientWise,\
        ClippingRennala, ClippingMindFlayer, FixedClipping, DynamicClippingRennala
from function import StochasticTridiagonalQuadraticFunction
from theoritical.probability_calculations import mf_calculate_ps, modmf_calculate_clipping_times
from theoritical.stepsizes import *
from theoritical.mf_theoritical_batches import MindFlayerBatchSize

CLASS_TO_NAME = {
        AsynchronousSGD.__name__: "ASGD",
        RennalaSGD.__name__: "Rennala",
        AsynchronousTimeFrameMiniBatchSGD.__name__: "MindFlayer",
        MindFlayerClientWise.__name__: "MindFlayer-CW",
        ClippingRennala.__name__: "ClippingRennala",
        ClippingMindFlayer.__name__: "ClippingMindFlayer",
        FixedClipping.__name__: "FixedClipping",
        DynamicClippingRennala.__name__: "DynamicClippingRennala"
}

def harmonia(scores, batch_size):
    length = len(scores)

    inf_indices = np.where(scores == -np.inf)[0]
    number_of_inf_indices = len(inf_indices)
    if number_of_inf_indices:
        ind = np.random.choice(inf_indices, min(batch_size, number_of_inf_indices), replace=False)
        allocation = np.zeros(length, dtype='int')
        allocation[ind] = 1
        scores[ind] = np.inf
        if batch_size <= number_of_inf_indices:
            return allocation
        return allocation + harmonia(scores=scores, batch_size=batch_size-number_of_inf_indices)

    if np.any(scores < 0):
        scores -= 2*np.min(scores)

    if batch_size == 1:
        best_allocation_index = np.random.choice(np.where(scores == np.min(scores))[0], 1)[0]
        return np.reshape(np.eye(length, dtype='int')[best_allocation_index], (length, ))

    best_prev_allocation = harmonia(scores=scores, batch_size=batch_size-1)
    allocations = np.tile(best_prev_allocation, (length, 1)) + np.eye(length, dtype=int)
    # Compute the corresponding times (max of scores * allocation)
    times = np.max(scores * allocations, axis=0)
    # Set the times for inf indices to np.inf
    times[scores == np.inf] = np.inf
    # Find a random index of the minimum time
    best_allocation_indices = np.where(times == np.min(times))[0]

    # choose the one with minimium cardinality
    best_allocation_index = min(
        best_allocation_indices, 
        key=lambda i: sum(scores * allocations[i] == np.min(times))
    )
    return allocations[best_allocation_index]



def extrapolate_results(times_list, grads_list, points_list):
    # Using numpy for efficient handling of operations
    all_times = np.unique(np.concatenate(times_list))

    # Initialize arrays to hold the interpolated values
    grad_arrays = []
    point_arrays = []

    for times, grads, points in zip(times_list, grads_list, points_list):
        # Create an indexer array to position each run's times into the all_times array
        indexer = np.searchsorted(all_times, times)

        # Initialize arrays for current run with NaNs which will later be forward-filled
        grad_array = np.full(all_times.shape, np.nan, dtype=np.float64)
        point_array = np.full(all_times.shape, np.nan, dtype=np.float64)

        # Place the grad and point values in their respective positions
        grad_array[indexer] = grads
        point_array[indexer] = points

        # Forward fill the NaN values
        invalid = np.isnan(grad_array)
        mask = np.where(~invalid, np.arange(len(grad_array)), 0)
        np.maximum.accumulate(mask, out=mask)

        grad_array = grad_array[mask]
        point_array = point_array[mask]

        grad_arrays.append(grad_array)
        point_arrays.append(point_array)

    # Ensuring correct DataFrame construction
    grad_df = pd.DataFrame(np.array(grad_arrays).T, index=all_times, columns=pd.Index(range(len(times_list)), name='run'))
    point_df = pd.DataFrame(np.array(point_arrays).T, index=all_times, columns=pd.Index(range(len(times_list)), name='run'))

    return pd.concat({'grad': grad_df, 'point': point_df}, axis=1)


def calculate_aggregations(df):
    # Example aggregation: mean, std, min, and max across runs for each measurement
    result = {}
    for measure in ['grad', 'point']:
        measure_df = df[measure]  # DataFrame slice with just the current measure
        result[measure] = {
            'mean': measure_df.mean(axis=1),
            'std': measure_df.std(axis=1),
            'min': measure_df.min(axis=1),
            'max': measure_df.max(axis=1)
        }

    return pd.concat({k: pd.DataFrame(v) for k, v in result.items()}, axis=1)


def create_exp_filename(config):
    name = CLASS_TO_NAME[config.get("optimizer_class").__name__]

    if name == "Rennala":
        batch_size = config.get("optimizer_params").get("batch_size")
        name += f"_b{batch_size}"
    elif name == "MindFlayer" or name == "MindFlayer-CW":
        T = config.get("optimizer_params").get("T")
        name += f"_T{T}"
    elif name == "FixedClipping":
        num_clips = config.get("optimizer_params").get("num_clips")
        batch_size = sum(num_clips)
        name += f"_b{batch_size}"
    elif name == "ClippingRennala":
        batch_size = config.get("optimizer_params").get("batch_size")
        prob = config.get("optimizer_params").get("p")
        name += f"_b{batch_size}_p{prob}"
    elif name == "DynamicClippingRennala":
        batch_size = config.get("optimizer_params").get("batch_size")
        prob = config.get("optimizer_params").get("p")
        name += f"_b{batch_size}_p{prob}"
    elif name == "ClippingMindFlayer":
        T = config.get("optimizer_params").get("T")
        clipping_time = config.get("optimizer_params").get("clipping_time")
        name += f"_T{T}_clip{clipping_time}"

    exp_num = config.get("exp_num")
    gamma = config.get("optimizer_params").get("gamma")
    dim = config.get("dim")
    num_nodes = config.get("num_nodes")
    filename = f"{exp_num}-{name}_gamma{gamma}_dim{dim}_nw{num_nodes}.npz"

    return filename


def configure_experiments(setup_config, random_time_setup, isTheoritical):
    experiments = []
    exp_num = 1

    for dim, num_nodes in product(setup_config.get("DIMS"), setup_config.get("NUM_NODES")):
        eps = setup_config.get("CONVERGENCE_THRESHOLD")
        sigma2 = setup_config.get("STOCH_GRAD_NOISE") ** 2 * dim

        main_diag, side_diag, b = create_worst_case(dim, 1)
        stochastic_func = StochasticTridiagonalQuadraticFunction(
            main_diag, side_diag, b, setup_config.get("function_seed"),
            setup_config.get("STOCH_GRAD_NOISE"), "add")

        random_time_setup["num_nodes"] = num_nodes


        # Theoritical experiments first
        if isTheoritical:
            if setup_config.get("do_ASGD", False):
                step_size = ASGDStepSize(stochastic_func, eps, sigma2)
                experiments.append({
                    "exp_num": exp_num,
                    "optimizer_class": AsynchronousSGD,
                    "optimizer_params": {"gamma": step_size},
                    "dim": dim,
                    "num_nodes": num_nodes,
                    "stochastic_func": stochastic_func
                })

                exp_num += 1

            if setup_config.get("do_FixedClipping", False):
                clipping_time = setup_config.get("CLIPPING")[0]
                clipping_times = [random_time_setup.get("func")(i) +\
                                   clipping_time for i in range(num_nodes)]
                ps = mf_calculate_ps(clipping_times, **random_time_setup)
                num_clips = MindFlayerBatchSize(ps, clipping_times, eps, sigma2)
                step_size = MindFlayerStepSize(stochastic_func, ps, num_clips, eps, sigma2)

                experiments.append({
                    "exp_num": exp_num,
                    "optimizer_class": FixedClipping,
                    "optimizer_params": {"gamma": step_size, "num_clips": num_clips,
                                         "clipping_times": clipping_times, "ps": ps,
                                         "clipping_time": clipping_time},
                    "dim": dim,
                    "num_nodes": num_nodes,
                    "stochastic_func": stochastic_func
                })

                exp_num += 1

            if setup_config.get("do_Rennala", False):
                batch_size = max(1, np.ceil(sigma2 / eps))
                step_size = RennalaStepSize(stochastic_func, batch_size, eps, sigma2)

                experiments.append({
                    "exp_num": exp_num,
                    "optimizer_class": AsynchronousMiniBatchSGD,
                    "optimizer_params": {"gamma": step_size, "batch_size": batch_size},
                    "dim": dim,
                    "num_nodes": num_nodes,
                    "stochastic_func": stochastic_func
                })

                exp_num += 1

            # For now try different p's and different B in theory
            for p in setup_config.get("PROB"):
                if setup_config.get("do_ClippingRennala", False):
                    clippings = modmf_calculate_clipping_times(p, **random_time_setup)
                    clipping_times = [random_time_setup.get("func")(i) +\
                                       clip for i, clip in enumerate(clippings)]

                    # This can potentially be changed but is set so that expected success = rennala batch size
                    batch_size = max(1, np.ceil(sigma2 / eps)) * 1/p
                    step_size = ModMindFlayerStepSize(stochastic_func, batch_size, p, eps, sigma2)

                    experiments.append({
                        "exp_num": exp_num,
                        "optimizer_class": ClippingRennala,
                        "optimizer_params": {"gamma": step_size,
                                             "clipping_times": clipping_times,
                                             "p": p,
                                             "batch_size": batch_size},
                        "dim": dim,
                        "num_nodes": num_nodes,
                        "stochastic_func": stochastic_func
                    })

                    exp_num += 1

                if setup_config.get("do_DynamicClippingRennala", False):
                    # This can potentially be changed but is set so that expected success = rennala batch size
                    batch_size = max(1, np.ceil(sigma2 / eps)) * 1/p
                    step_size = ModMindFlayerStepSize(stochastic_func, batch_size, p, eps, sigma2)

                    experiments.append({
                        "exp_num": exp_num,
                        "optimizer_class": DynamicClippingRennala,
                        "optimizer_params": {"gamma": step_size,
                                             "p": p,
                                             "batch_size": batch_size},
                        "dim": dim,
                        "num_nodes": num_nodes,
                        "stochastic_func": stochastic_func
                    })

                    exp_num += 1

        else:
            for step_size in setup_config.get("STEP_SIZES"):
                if setup_config.get("do_ASGD", False):
                    experiments.append({
                        "exp_num": exp_num,
                        "optimizer_class": AsynchronousSGD,
                        "optimizer_params": {"gamma": step_size},
                        "dim": dim,
                        "num_nodes": num_nodes,
                        "stochastic_func": stochastic_func
                    })

                    exp_num += 1

                if setup_config.get("do_FixedClipping", False):
                    for clipping_time in setup_config.get("CLIPPING"):
                        clipping_times = [random_time_setup.get("func")(i) +\
                                          clipping_time for i in range(num_nodes)]
                        ps = mf_calculate_ps(clipping_times, **random_time_setup)
                        num_clips = MindFlayerBatchSize(ps, clipping_times, eps, sigma2)

                        experiments.append({
                            "exp_num": exp_num,
                            "optimizer_class": FixedClipping,
                            "optimizer_params": {"gamma": step_size, "num_clips": num_clips,
                                                 "clipping_times": clipping_times, "ps": ps},
                            "dim": dim,
                            "num_nodes": num_nodes,
                            "stochastic_func": stochastic_func
                        })

                        exp_num += 1

                for batch_size in setup_config.get("BATCH_SIZES"):
                    if setup_config.get("do_Rennala", False):
                        experiments.append({
                            "exp_num": exp_num,
                            "optimizer_class": AsynchronousMiniBatchSGD,
                            "optimizer_params": {"gamma": step_size, "batch_size": batch_size},
                            "dim": dim,
                            "num_nodes": num_nodes,
                            "stochastic_func": stochastic_func
                        })

                        exp_num += 1

                    if setup_config.get("do_ClippingRennala", False):
                        clipping_times = [random_time_setup.get("func")(i) +\
                                          clipping_time for i in range(num_nodes)]
                        ps = mf_calculate_ps(clipping_times, **random_time_setup)

                        experiments.append({
                            "exp_num": exp_num,
                            "optimizer_class": ClippingRennala,
                            "optimizer_params": {"gamma": step_size,
                                                 "clipping_times": clipping_times,
                                                 "p": ps[0], 
                                                 "batch_size": batch_size},
                            "dim": dim,
                            "num_nodes": num_nodes,
                            "stochastic_func": stochastic_func
                        })

                        exp_num += 1

        return experiments

def time_setup(generator, **kwargs):
    delay_type = kwargs.get("type")
    delay_func = kwargs.get("func")
    num_nodes = kwargs.get("num_nodes")

    delays = [delay_func(i) for i in range(num_nodes)]

    if delay_type == "exponential":
        scale = kwargs.get("scale")
        noise_function = lambda *args: generator.exponential(scale=scale)
    elif delay_type == "inf_bernoulli":
        prob = kwargs.get("p")
        inv_dirac = lambda x: 0 if x == 1 else np.inf
        noise_function = lambda *args: inv_dirac(generator.binomial(n=1, p=prob))
    elif delay_type == "alpha_bernoulli":
        prob = kwargs.get("p")
        alpha = kwargs.get("alpha")
        noise_function = lambda *args: generator.binomial(n=1, p=1-prob) * (alpha - 1) * args[0]
    elif delay_type == "cauchy":
        scale = kwargs.get("scale")
        loc = kwargs.get("loc")
        noise_function = lambda *args: np.abs(sps.cauchy(scale=scale, loc=loc).rvs(random_state=generator))
    elif delay_type == "levy":
        scale = kwargs.get("scale")
        loc = kwargs.get("loc")
        noise_function = lambda *args: sps.levy(scale=scale, loc=loc).rvs(random_state=generator)
    elif delay_type == "levy_exp_bernoulli":
        scale1 = kwargs.get("scale1")
        scale2 = kwargs.get("scale2")
        loc2 = kwargs.get("loc2")
        prob = kwargs.get("p")
        noise_function = lambda *args: generator.exponential(scale=scale1)\
                if generator.binomial(n=1, p=prob) == 1 else\
           sps.levy(scale=scale2, loc=loc2).rvs(random_state=generator)
    elif delay_type == "lognorm":
        s = kwargs.get("s")
        noise_function = lambda *args: sps.lognorm(s).rvs(random_state=generator)
    elif delay_type == "heterolognorm":
        s = kwargs.get("s")
        het_func = kwargs.get("het_func")
        noise_function = lambda *args: sps.lognorm(s * het_func(args[0])).rvs(random_state=generator)
    else:
        raise ValueError(f"delay type not supported: {delay_type}")

    return delays, noise_function
