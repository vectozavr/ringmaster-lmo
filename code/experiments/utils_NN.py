import numpy as np
import pandas as pd
import scipy.stats as sps


import torch
import torchvision
import torchvision.transforms as transforms

from function import parameters_to_tensor, tensor_to_parameters, NeuralNetworkFunction, BaseStochasticFunction
from asynchronous.algorithm import AsynchronousSGD, FixedClipping
from theoritical.probability_calculations import calculate_p0,\
        calculate_p0_tilde, mf_calculate_ps
from theoritical.mf_theoritical_batches import MindFlayerBatchSize

CLASS_TO_NAME = {
        AsynchronousSGD.__name__: "ASGD",
        FixedClipping.__name__: "FixedClipping"
}


# def prepare_dataset(path_to_dataset, number_of_first_examples=None, dataset='mnist', is_cuda=False):
#     if dataset == 'mnist':
#         transform_train = [transforms.ToTensor(),
#                            transforms.Normalize(mean=0.1307, std=0.3081)]
#     else:
#         raise RuntimeError()

#     transform_train = transforms.Compose(transform_train)
#     target_transform = None

#     if dataset == 'mnist':
#         trainset = torchvision.datasets.MNIST(root=path_to_dataset, train=True, download=True,
#                                                 transform=transform_train,
#                                                 target_transform=target_transform)
#     else:
#         raise RuntimeError()


#     if number_of_first_examples is not None:
#         trainset = torch.utils.data.Subset(trainset, range(number_of_first_examples))


#     dataloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
#     features, labels = next(iter(dataloader))

#     if features.dim() > 2:
#         features = features.view(features.size(0), -1)

#     sample = features[0]

#     stats_labels = np.unique(labels.cpu().numpy(), return_counts=True)
#     print(f"Stats Labels: {stats_labels}")
#     number_of_classes = len(np.unique(labels))
#     print(f"Number of classes: {number_of_classes}")

#     return features.numpy(), labels.numpy(), number_of_classes



def prepare_dataset(path_to_dataset, number_of_first_examples=None, dataset='mnist', is_cuda=False):
    # Per-dataset normalization stats (standard choices used in practice)
    if dataset == 'mnist':
        mean, std = 0.1307, 0.3081
    elif dataset == 'fashion':
        # Fashion-MNIST (Zalando) commonly used stats
        mean, std = 0.2860406, 0.35302424
    else:
        raise RuntimeError(f"Unknown dataset '{dataset}', expected 'mnist' or 'fashion'.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    target_transform = None

    # Pick dataset class
    if dataset == 'mnist':
        ds_cls = torchvision.datasets.MNIST
        ds_kwargs = dict(root=path_to_dataset, train=True, download=True,
                         transform=transform, target_transform=target_transform)
    elif dataset == 'fashion':
        ds_cls = torchvision.datasets.FashionMNIST
        ds_kwargs = dict(root=path_to_dataset, train=True, download=True,
                         transform=transform, target_transform=target_transform)

    trainset = ds_cls(**ds_kwargs)

    # Optional subset of the first N examples (keeps your original behavior)
    if number_of_first_examples is not None:
        trainset = torch.utils.data.Subset(trainset, range(number_of_first_examples))

    # Load everything in one go (you were already doing this)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    features, labels = next(iter(dataloader))  # features: [N, 1, 28, 28], labels: [N]

    # Flatten to [N, 784] to match your SimpleNeuralNetFunction expectations
    if features.dim() > 2:
        features = features.view(features.size(0), -1)

    # Basic stats/prints (unchanged)
    stats_labels = np.unique(labels.cpu().numpy(), return_counts=True)
    print(f"Stats Labels: {stats_labels}")
    number_of_classes = len(np.unique(labels))
    print(f"Number of classes: {number_of_classes}")

    return features.numpy(), labels.numpy(), number_of_classes



class StochasticNeuralNetworkFunction(BaseStochasticFunction):
    def __init__(self, features, labels, number_of_classes=2, is_cuda=False,
                 reg_parameter=0.0, neural_network_name=None, noise=0.1, seed=None):
        super().__init__(seed)
        self.nn_function = NeuralNetworkFunction(features, labels,
                                                 number_of_classes, is_cuda,
                                                 reg_parameter,
                                                 neural_network_name)
        self._noise = noise

    def value(self, point):
        return self.nn_function.value(point)

    def gradient(self, point):
        return self.nn_function.gradient(point)

    def dim(self):
        return self.nn_function.dim()

    def get_current_point(self):
        return self.nn_function.get_current_point()

    def _loss(self, point):
        return self.nn_function._loss(point())

    def _logits(self, point):
        return self.nn_function._logits(point)

    def _check_accuracy(self, point):
        return self.nn_function._check_accuracy(point)

    def stochastic_gradient_at_points(self, points):
        gradients = []
        for point in points:
            deterministic_grad = self.nn_function.gradient(point)
            noise = self._noise * self._generator.normal()
            noisy_grad = deterministic_grad + noise
            gradients.append(noisy_grad)
        return gradients


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
    elif name == "FixedClipping":
        num_clips = config.get("optimizer_params").get("num_clips")
        clipping_time = config.get("optimizer_params").get("clipping_time")
        name += f"_bfixed{num_clips}_clip{clipping_time}"

    gamma = config.get("optimizer_params").get("gamma")
    num_nodes = config.get("num_nodes")
    filename = f"{name}_gamma{gamma}_nw{num_nodes}.npz"

    return filename

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
    elif delay_type == "logcauchy":
        scale = kwargs.get("scale")
        loc = kwargs.get("loc")
        noise_function = lambda *args: np.exp(sps.cauchy(scale=scale, loc=loc).rvs(random_state=generator))
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
    else:
        raise ValueError(f"delay type not supported: {delay_type}")

    return delays, noise_function


def configure_experiments(setup_config, random_time_setup):
    experiments = []
    exp_num = 1

    for num_nodes in setup_config.get("NUM_NODES"):
        eps = setup_config.get("CONVERGENCE_THRESHOLD")

        features, labels, number_of_classes = prepare_dataset("./", 1000)
        stochastic_func = StochasticNeuralNetworkFunction(features, labels, 10,
                                                          neural_network_name="two_layer_neural_net_relu",
                                                          noise=setup_config.get("STOCH_GRAD_NOISE"),
                                                          seed=setup_config.get("function_seed"),
                                                          is_cuda=torch.cuda.is_available())

        sigma2 = setup_config.get("STOCH_GRAD_NOISE") ** 2 * stochastic_func.dim()


        for step_size in setup_config.get("STEP_SIZES"):
            if setup_config.get("do_ASGD", False):
                experiments.append({
                    "exp_num": exp_num,
                    "optimizer_class": AsynchronousSGD,
                    "optimizer_params": {"gamma": step_size},
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
                                             "clipping_times": clipping_times, "ps": ps,
                                             "clipping_time": clipping_time},
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
                        "num_nodes": num_nodes,
                        "stochastic_func": stochastic_func
                    })

                    exp_num += 1

    return experiments
