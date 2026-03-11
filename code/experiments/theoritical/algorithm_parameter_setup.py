import numpy as np
import scipy.stats as sps
from itertools import product
import pandas as pd

from distributed_optimization_library.function import create_worst_case
from distributed_optimization_library.asynchronous.algorithm import\
        AsynchronousSGD, AsynchronousMiniBatchSGD, MiniBatchSGD,\
        AsynchronousTimeFrameMiniBatchSGD, MindFlayerClientWise,\
        ClippingRennala, ClippingMindFlayer, FixedClipping
from distributed_optimization_library.function import StochasticTridiagonalQuadraticFunction
from distributed_optimization_library.theoritical.probability_calculations import\
        calculate_p0, calculate_p0_tilde, mf_calculate_ps
from distributed_optimization_library.theoritical.stepsizes import\
        RennalaStepSize, MindFlayerStepSize, ASGDStepSize
from distributed_optimization_library.theoritical.mf_theoritical_batches import\
        MindFlayerBatchSize


asgd_exp = setup_ASGD()
asgd_exp["exp_num"] = exp_num
asgd_exp["stochastic_func"] = stochastic_func
experiments.append(asgd_exp)


def setup_ASGD(dim, num_nodes, step_size, setup_config):
    eps = setup_config.get("CONVERGENCE_THRESHOLD")
    sigma2 = setup_config.get("STOCH_GRAD_NOISE") ** 2 * dim

    main_diag, side_diag, b = create_worst_case(dim, 1)
    stochastic_func = StochasticTridiagonalQuadraticFunction(
            main_diag, side_diag, b, setup_config.get("function_seed"),
            setup_config.get("STOCH_GRAD_NOISE"), "add")

    if not step_size:
        step_size = ASGDStepSize(stochastic_func, eps, sigma2)

    return {
        "optimizer_class": AsynchronousSGD,
        "optimizer_params": {"gamma": step_size},
        "dim": dim,
        "num_nodes": num_nodes,
    }

def setup_MindFlayer(dim, num_nodes, step_size, setup_config, random_time_setup):
    clipping_time = setup_config.get("CLIPPING")[0]
    clipping_times = [random_time_setup.get("func")(i) +\
                       clipping_time for i in range(num_nodes)]
    ps = mf_calculate_ps(clipping_times, **random_time_setup)
    num_clips = MindFlayerBatchSize(ps, clipping_times, eps, sigma2)
    step_size = MindFlayerStepSize(stochastic_func, ps, num_clips, eps, sigma2)

    return {
        "optimizer_class": FixedClipping,
        "optimizer_params": {"gamma": step_size, "num_clips": num_clips,
                             "clipping_times": clipping_times, "ps": ps,
                             "clipping_time": clipping_time},
        "dim": dim,
        "num_nodes": num_nodes
    }
