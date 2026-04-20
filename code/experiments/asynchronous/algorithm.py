import heapq
import numpy as np
from factory import Factory
from function import OptimizationProblemMeta
from signature import Signature
from asynchronous.asynchronous_transport import DelayedAsynchronousTransport, RandomDelayedAsynchronousTransport
from concurrent.futures import ProcessPoolExecutor


def _zeropower_via_newtonschulz5_numpy(gradient_matrix, steps):
    """Approximate Muon's orthogonalization step with the Newton-Schulz iteration."""
    assert gradient_matrix.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    update = np.array(gradient_matrix, dtype=np.float64, copy=True)
    transposed = False
    if update.shape[0] > update.shape[1]:
        update = update.T
        transposed = True

    update /= np.linalg.norm(update) + 1e-7
    for _ in range(steps):
        gram = update @ update.T
        update = a * update + (b * gram + c * gram @ gram) @ update

    if transposed:
        update = update.T
    return update


def _muon_update_numpy(gradient, momentum, beta=0.95, ns_steps=5, nesterov=True):
    """Return Muon's orthogonalized update along with the updated momentum buffer.

    In this repository the optimized point is a single vector, so 1D updates are treated
    as 1 x d matrices before the Muon orthogonalization step.
    """
    next_momentum = beta * momentum + (1.0 - beta) * gradient
    if nesterov:
        update = beta * next_momentum + (1.0 - beta) * gradient
    else:
        update = next_momentum

    original_shape = update.shape
    if update.ndim == 1:
        matrix_update = update.reshape(1, -1)
    elif update.ndim == 2:
        matrix_update = update
    else:
        matrix_update = update.reshape(update.shape[0], -1)

    matrix_update = _zeropower_via_newtonschulz5_numpy(matrix_update, steps=ns_steps)
    matrix_update *= np.sqrt(max(1.0, matrix_update.shape[0] / matrix_update.shape[1]))
    return matrix_update.reshape(original_shape), next_momentum


def _momentum_update_numpy(gradient, momentum, beta=0.95, nesterov=True):
    """Momentum fallback for parameters where Muon should not be applied."""
    next_momentum = beta * momentum + (1.0 - beta) * gradient
    if nesterov:
        update = beta * next_momentum + (1.0 - beta) * gradient
    else:
        update = next_momentum
    return update, next_momentum


def _build_parameter_infos(meta, point_dim):
    if not isinstance(meta, dict):
        return None
    raw_infos = meta.get("parameter_infos")
    if not raw_infos:
        return None

    parameter_infos = []
    total_size = 0
    for info in raw_infos:
        shape = tuple(info["shape"])
        size = int(np.prod(shape))
        parameter_infos.append({
            "shape": shape,
            "size": size,
            "use_muon": bool(info.get("use_muon", len(shape) >= 2)),
        })
        total_size += size

    if total_size != point_dim:
        raise ValueError(f"Parameter metadata size mismatch: expected {point_dim}, got {total_size}")
    return parameter_infos


def _structured_muon_update_numpy(gradient, momentum, parameter_infos, beta=0.95, ns_steps=5, nesterov=True):
    """Apply Muon blockwise to structured parameters and momentum-SGD elsewhere."""
    update = np.zeros_like(gradient, dtype=np.float64)
    next_momentum = np.zeros_like(momentum, dtype=np.float64)
    shift = 0
    for info in parameter_infos:
        size = info["size"]
        shape = info["shape"]
        grad_block = gradient[shift:shift + size].reshape(shape)
        momentum_block = momentum[shift:shift + size].reshape(shape)
        if info["use_muon"] and len(shape) >= 2:
            update_block, next_momentum_block = _muon_update_numpy(
                grad_block,
                momentum_block,
                beta=beta,
                ns_steps=ns_steps,
                nesterov=nesterov,
            )
        else:
            update_block, next_momentum_block = _momentum_update_numpy(
                grad_block,
                momentum_block,
                beta=beta,
                nesterov=nesterov,
            )
        update[shift:shift + size] = update_block.reshape(-1)
        next_momentum[shift:shift + size] = next_momentum_block.reshape(-1)
        shift += size
    return update, next_momentum


def _refresh_ringmaster_workers(transport, heap, time, point, delays, threshold, iteration):
    if np.max(delays) < threshold:
        return heap

    indices = np.where(delays >= threshold)[0]
    delays[indices] = 0
    heap = [item for item in heap if item[1] not in indices]
    heapq.heapify(heap)

    for node_index in indices:
        transport.ignore_node(time, node_index)
        available_time = transport.call_available_node_method(
            time, node_index, node_method="calculate_stochastic_gradient", point=point)
        heapq.heappush(heap, (available_time, node_index, iteration))

    return heap

class FactoryAsyncMaster(Factory):
    pass

class FactoryAsyncNode(Factory):
    pass


class StochasticGradientNodeAlgorithm(object):
    def __init__(self, function, **kwargs):
        self._function = function
    
    def calculate_stochastic_gradient(self, point):
        return self._function.stochastic_gradient(point)
    
    def calculate_function(self, point):
        return self._function.value(point)
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)


@FactoryAsyncNode.register("asynchronous_sgd_node")
class AsynchronousSGDNode(StochasticGradientNodeAlgorithm):
    pass


@FactoryAsyncNode.register("rennala_node")
class AsynchronousMiniBatchSGDNode(StochasticGradientNodeAlgorithm):
    pass


@FactoryAsyncNode.register("minibatch_sgd_node")
class MiniBatchSGDNode(StochasticGradientNodeAlgorithm):
    pass

@FactoryAsyncMaster.register("Ringmaster")
class RingmasterASGD(object):
    def __init__(self, transport, point, max_delay, gamma=None, gamma_multiply=None, seed=None, meta=None):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._max_delay = max_delay
        self._seed = seed
        self._time = 0
        
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._delays = np.array([0] * self._number_of_nodes)
        
        for node_index in range(self._transport.get_number_of_nodes()):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            heapq.heappush(self._heap, (available_time, node_index, self._iter))
    
    def step(self):
        available_time, node_index, iter = heapq.heappop(self._heap)
        assert available_time != np.inf
        self._time = available_time
        stochastic_gradient = self._transport.call_ready_node(self._time, node_index)

        self._point = self._point - self._gamma * stochastic_gradient
        self._iter += 1
        available_time = self._transport.call_available_node_method(
            self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
        heapq.heappush(self._heap, (available_time, node_index, self._iter))

        self._delays += 1 # increase the delay by 1 for all workers
        self._delays[node_index] = 0
        if np.max(self._delays) == self._max_delay:
            # print("Delays: ", self._delays)
            # print("Heap before deletion: ", self._heap)
            indices = np.where(self._delays == self._max_delay)[0]
            self._delays[indices] = 0
            self._heap = [item for item in self._heap if item[1] not in indices]
            heapq.heapify(self._heap)
            # print("Heap after deletion: ", self._heap)

            for node_index in indices:
                self._transport.ignore_node(self._time, node_index)
                available_time = self._transport.call_available_node_method(
                    self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
                heapq.heappush(self._heap, (available_time, node_index, self._iter))

            # print("Heap after pushing again: ", self._heap)

    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
        
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time


@FactoryAsyncMaster.register("RingmasterMuon")
class RingmasterMuonASGD(object):
    def __init__(
        self,
        transport,
        point,
        max_delay,
        gamma=None,
        gamma_multiply=None,
        beta=0.95,
        ns_steps=5,
        nesterov=True,
        seed=None,
        meta=None,
    ):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._max_delay = max_delay
        self._beta = beta
        self._ns_steps = ns_steps
        self._nesterov = nesterov
        self._seed = seed
        self._time = 0

        self._momentum = np.zeros_like(self._point, dtype=np.float64)
        self._parameter_infos = _build_parameter_infos(meta, self._point.size)
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._delays = np.array([0] * self._number_of_nodes)

        for node_index in range(self._transport.get_number_of_nodes()):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            heapq.heappush(self._heap, (available_time, node_index, self._iter))

    def step(self):
        available_time, node_index, iter = heapq.heappop(self._heap)
        assert available_time != np.inf
        self._time = available_time
        stochastic_gradient = self._transport.call_ready_node(self._time, node_index)

        if self._parameter_infos is None:
            muon_update, self._momentum = _muon_update_numpy(
                stochastic_gradient,
                self._momentum,
                beta=self._beta,
                ns_steps=self._ns_steps,
                nesterov=self._nesterov,
            )
        else:
            muon_update, self._momentum = _structured_muon_update_numpy(
                stochastic_gradient,
                self._momentum,
                self._parameter_infos,
                beta=self._beta,
                ns_steps=self._ns_steps,
                nesterov=self._nesterov,
            )
        self._point = self._point - self._gamma * muon_update
        self._iter += 1
        available_time = self._transport.call_available_node_method(
            self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
        heapq.heappush(self._heap, (available_time, node_index, self._iter))

        self._delays += 1
        self._delays[node_index] = 0
        if np.max(self._delays) == self._max_delay:
            indices = np.where(self._delays == self._max_delay)[0]
            self._delays[indices] = 0
            self._heap = [item for item in self._heap if item[1] not in indices]
            heapq.heapify(self._heap)

            for node_index in indices:
                self._transport.ignore_node(self._time, node_index)
                available_time = self._transport.call_available_node_method(
                    self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
                heapq.heappush(self._heap, (available_time, node_index, self._iter))

    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))

    def get_point(self):
        return self._point

    def get_time(self):
        return self._time


@FactoryAsyncMaster.register("ParameterAgnosticRingmasterMuon")
class ParameterAgnosticRingmasterMuonASGD(object):
    def __init__(self, transport, point, eta=None, ns_steps=5, nesterov=True, seed=None, meta=None):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        self._eta = eta
        self._ns_steps = ns_steps
        self._nesterov = nesterov
        self._seed = seed
        self._time = 0

        self._alpha = 1.0
        self._threshold = 1.0
        self._momentum = np.zeros_like(self._point, dtype=np.float64)
        self._parameter_infos = _build_parameter_infos(meta, self._point.size)
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._delays = np.array([0] * self._number_of_nodes)

        for node_index in range(self._transport.get_number_of_nodes()):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            heapq.heappush(self._heap, (available_time, node_index, self._iter))

    def step(self):
        available_time, node_index, iter = heapq.heappop(self._heap)
        assert available_time != np.inf
        self._time = available_time
        stochastic_gradient = self._transport.call_ready_node(self._time, node_index)

        stepsize = self._eta / ((self._iter + 1) ** 0.75)
        beta = 1.0 - self._alpha
        if self._parameter_infos is None:
            muon_update, self._momentum = _muon_update_numpy(
                stochastic_gradient,
                self._momentum,
                beta=beta,
                ns_steps=self._ns_steps,
                nesterov=self._nesterov,
            )
        else:
            muon_update, self._momentum = _structured_muon_update_numpy(
                stochastic_gradient,
                self._momentum,
                self._parameter_infos,
                beta=beta,
                ns_steps=self._ns_steps,
                nesterov=self._nesterov,
            )
        self._point = self._point - stepsize * muon_update

        self._iter += 1
        available_time = self._transport.call_available_node_method(
            self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
        heapq.heappush(self._heap, (available_time, node_index, self._iter))

        self._delays += 1
        self._delays[node_index] = 0
        self._heap = _refresh_ringmaster_workers(
            self._transport,
            self._heap,
            self._time,
            self._point,
            self._delays,
            self._threshold,
            self._iter,
        )

        self._alpha = 1.0 / np.sqrt(self._iter)
        self._threshold = max(1.0, np.floor(1.0 / self._alpha))

    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))

    def get_point(self):
        return self._point

    def get_time(self):
        return self._time

@FactoryAsyncMaster.register("Momentum_Normalized_Ringmaster")
class Momentum_Normalized_RingmasterASGD(object):
    def __init__(self, transport, point, parameter_agnostic=False, threshold=None, gamma=None, alpha=None, seed=None):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        self._parameter_agnostic = parameter_agnostic
        self._gamma = gamma
        if parameter_agnostic:
            self._threshold = 1
            self._alpha = 1
        else:
            self._threshold = threshold
            self._alpha = alpha
        self._seed = seed
        self._time = 0
        
        self._momentum = 0.0
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._delays = np.array([0] * self._number_of_nodes)
        
        for node_index in range(self._transport.get_number_of_nodes()):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            heapq.heappush(self._heap, (available_time, node_index, self._iter))
    
    def step(self):
        available_time, node_index, iter = heapq.heappop(self._heap)
        assert available_time != np.inf
        self._time = available_time

        stepsize = self._gamma / ( (self._iter+1) ** 0.75 ) if self._parameter_agnostic else self._gamma

        stochastic_gradient = self._transport.call_ready_node(self._time, node_index)
        # self._momentum = self._alpha**self._delays[node_index] * stochastic_gradient + (1-self._alpha) * self._momentum
        self._momentum = self._alpha * stochastic_gradient + (1-self._alpha) * self._momentum
        self._point = self._point - stepsize * ( self._momentum / ( np.linalg.norm(self._momentum) + 1e-10 ) )

        self._iter += 1
        available_time = self._transport.call_available_node_method(
            self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
        heapq.heappush(self._heap, (available_time, node_index, self._iter))

        self._delays += 1 # increase the delay by 1 for all workers
        self._delays[node_index] = 0
        if np.max(self._delays) >= self._threshold:
            indices = np.where(self._delays >= self._threshold)[0]
            self._delays[indices] = 0
            self._heap = [item for item in self._heap if item[1] not in indices]
            heapq.heapify(self._heap)

            for node_index in indices:
                self._transport.ignore_node(self._time, node_index)
                available_time = self._transport.call_available_node_method(
                    self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
                heapq.heappush(self._heap, (available_time, node_index, self._iter))
        
        if self._parameter_agnostic:
            # self._alpha = max(0.01, 1 / np.sqrt(self._iter))
            self._alpha = 1 / np.sqrt(self._iter)
            self._threshold = np.floor(np.sqrt(self._iter))
        print(self._threshold)

    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
        
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time

@FactoryAsyncMaster.register("asynchronous_sgd_master")
class AsynchronousSGD(object):
    def __init__(self, transport, point, gamma=None, delay_adaptive=False, gamma_multiply=None, seed=None, meta=None):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._delay_adaptive = delay_adaptive
        self._seed = seed
        self._time = 0
        
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._delays = np.array([0] * self._number_of_nodes)

        
        for node_index in range(self._transport.get_number_of_nodes()):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            heapq.heappush(self._heap, (available_time, node_index, self._iter))
    
    def step(self):
        available_time, node_index, iter = heapq.heappop(self._heap)
        assert available_time != np.inf
        self._time = available_time
        stochastic_gradient = self._transport.call_ready_node(self._time, node_index)

        if self._delay_adaptive:
            lr = self._gamma * self._number_of_nodes / max(self._number_of_nodes, self._delays[node_index])
            self._point = self._point - lr * stochastic_gradient
        else:
            self._point = self._point - self._gamma * stochastic_gradient

        self._iter += 1
        available_time = self._transport.call_available_node_method(
            self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
        heapq.heappush(self._heap, (available_time, node_index, self._iter))

        self._delays += 1
        self._delays[node_index] = 0

        
    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
        
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time


@FactoryAsyncMaster.register("AsynchronousMuon")
class DelayAdaptiveMuonASGD(object):
    def __init__(
        self,
        transport,
        point,
        gamma=None,
        delay_adaptive=False,
        gamma_multiply=None,
        beta=0.95,
        ns_steps=5,
        nesterov=True,
        seed=None,
        meta=None,
    ):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._delay_adaptive = delay_adaptive
        self._beta = beta
        self._ns_steps = ns_steps
        self._nesterov = nesterov
        self._seed = seed
        self._time = 0

        self._momentum = np.zeros_like(self._point, dtype=np.float64)
        self._parameter_infos = _build_parameter_infos(meta, self._point.size)
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._delays = np.array([0] * self._number_of_nodes)

        for node_index in range(self._transport.get_number_of_nodes()):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            heapq.heappush(self._heap, (available_time, node_index, self._iter))

    def step(self):
        available_time, node_index, iter = heapq.heappop(self._heap)
        assert available_time != np.inf
        self._time = available_time
        stochastic_gradient = self._transport.call_ready_node(self._time, node_index)

        if self._parameter_infos is None:
            muon_update, self._momentum = _muon_update_numpy(
                stochastic_gradient,
                self._momentum,
                beta=self._beta,
                ns_steps=self._ns_steps,
                nesterov=self._nesterov,
            )
        else:
            muon_update, self._momentum = _structured_muon_update_numpy(
                stochastic_gradient,
                self._momentum,
                self._parameter_infos,
                beta=self._beta,
                ns_steps=self._ns_steps,
                nesterov=self._nesterov,
            )
        if self._delay_adaptive:
            lr = self._gamma * self._number_of_nodes / max(self._number_of_nodes, self._delays[node_index])
        else:
            lr = self._gamma
        self._point = self._point - lr * muon_update

        self._iter += 1
        available_time = self._transport.call_available_node_method(
            self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
        heapq.heappush(self._heap, (available_time, node_index, self._iter))

        self._delays += 1
        self._delays[node_index] = 0

    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))

    def get_point(self):
        return self._point

    def get_time(self):
        return self._time

@FactoryAsyncMaster.register("IAASGD")
class IAASGD(object):
    def __init__(self, transport, point, gamma=None, delay_adaptive=False, gamma_multiply=None, seed=None, meta=None):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._delay_adaptive = delay_adaptive
        self._seed = seed
        self._time = 0
        
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._delays = np.array([0] * self._number_of_nodes)
        
        for node_index in range(self._transport.get_number_of_nodes()):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            heapq.heappush(self._heap, (available_time, node_index, self._iter))

        self._gradient_table = [None] * self._number_of_nodes
        self._valid_count = 0
        self._table_full = False
        self._mean_grad = np.zeros_like(self._point)

    # NEW: allow multiple arrivals per call (default = number of workers)
    def step(self, max_events=None):
        n = self._number_of_nodes
        K = n if (max_events is None) else int(max_events)
        if K <= 0:
            return

        # --- Warm-up: fill the table in a single call (no early returns) ---
        if not self._table_full:
            while self._valid_count < n:
                available_time, node_index, _ = heapq.heappop(self._heap)
                self._time = available_time

                g = self._transport.call_ready_node(self._time, node_index)
                g = np.asarray(g, dtype=self._point.dtype)

                old = self._gradient_table[node_index]
                if old is None:
                    self._gradient_table[node_index] = g
                    self._valid_count += 1
                else:
                    # if it happens during warm-up, just replace
                    self._gradient_table[node_index] = g

                next_time = self._transport.call_available_node_method(
                    self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
                heapq.heappush(self._heap, (next_time, node_index, self._iter))

            # table just became full: compute initial mean once
            G = np.stack(self._gradient_table, axis=0)   # (n, d)
            self._mean_grad = G.mean(axis=0)             # (d,)
            self._table_full = True
            # fall through to steady-state below

        # --- Steady-state: process up to K arrivals this call ---
        for _ in range(K):
            available_time, node_index, _ = heapq.heappop(self._heap)
            self._time = available_time

            g = self._transport.call_ready_node(self._time, node_index)
            g = np.asarray(g, dtype=self._point.dtype)

            old = self._gradient_table[node_index]
            if old is None:
                # should not happen after warm-up, but keep it safe
                self._gradient_table[node_index] = g
                # recompute mean conservatively once if needed
                G = np.stack([x if x is not None else g for x in self._gradient_table], axis=0)
                self._mean_grad = G.mean(axis=0)
            else:
                # Incremental mean update: m <- m + (g - old)/n
                np.add(self._mean_grad, (g - old) / float(n), out=self._mean_grad)
                self._gradient_table[node_index] = g

            # One async-SGD update per arrival
            self._point -= self._gamma * self._mean_grad
            self._iter += 1

            next_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            heapq.heappush(self._heap, (next_time, node_index, self._iter))

    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
        
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time

@FactoryAsyncMaster.register("Ringleader")
class RingleaderASGD(object):
    """
    Phase 1: collect until every worker has contributed at least once (no updates).
    Phase 2: perform exactly n updates — start with the last finisher of Phase 1.
             Extra arrivals from already-updated workers go into NEXT buffer.
    After Phase 2: move NEXT -> CURRENT, clear NEXT, and start a new Phase 1.
    """
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, meta=None):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._time = 0
        self._iter = 0

        self._number_of_nodes = n = self._transport.get_number_of_nodes()
        d = int(np.asarray(self._point).size)

        # CURRENT table (running per-node means at the current point)
        self._mean_curr   = np.zeros((n, d), dtype=self._point.dtype)
        self._cnt_curr    = np.zeros(n, dtype=np.int64)
        self._global_mean = np.zeros(d, dtype=self._point.dtype)

        # NEXT buffer (arrivals during Phase 2 from already-updated workers)
        self._mean_next  = np.zeros((n, d), dtype=self._point.dtype)
        self._cnt_next   = np.zeros(n, dtype=np.int64)
        self._global_mean_next = np.zeros(d, dtype=self._point.dtype)

        # Phase-1 bookkeeping
        self._have_arrived  = np.zeros(n, dtype=np.int8)  # whether worker i has contributed in current cycle
        self._arrived_count = 0

        # Event queue
        self._heap = []
        for i in range(n):
            tnext = self._transport.call_available_node_method(
                self._time, i, node_method="calculate_stochastic_gradient", point=self._point
            )
            heapq.heappush(self._heap, (tnext, i, self._iter))

    # ---- helpers ----
    def _asarray_like_point(self, g):
        # Ensure 1-D shape compatible with rows of mean tables
        return np.asarray(g, dtype=self._point.dtype).ravel()

    @staticmethod
    def _inc_mean(mean_mat, cnt_arr, i, g, n_for_global, global_vec):
        """
        Incremental mean for worker i and corresponding global mean:
          mean_i <- mean_i + (g - mean_i)/(cnt_i + 1)
          global <- global + (mean_i_new - mean_i_old)/n
        """
        c_new = int(cnt_arr[i]) + 1
        cnt_arr[i] = c_new
        mi = mean_mat[i]                 # view (1d)
        delta = (g - mi) / float(c_new)  # (d,)
        np.add(mi, delta, out=mi)
        np.add(global_vec, delta / float(n_for_global), out=global_vec)

    def _finish_cycle_and_prepare_next(self):
        """Promote NEXT -> CURRENT (O(1) swaps), clear NEXT, rebuild Phase-1 flags."""
        # Swap tables/globals
        self._mean_curr,  self._mean_next  = self._mean_next,  self._mean_curr
        self._cnt_curr,   self._cnt_next   = self._cnt_next,   self._cnt_curr
        self._global_mean, self._global_mean_next = self._global_mean_next, self._global_mean
        # Clear NEXT in-place
        self._mean_next.fill(0.0)
        self._cnt_next.fill(0)
        self._global_mean_next.fill(0.0)
        # Phase 1 pre-arrivals (fast workers may already be nonzero)
        self._have_arrived[:] = (self._cnt_curr > 0).astype(np.int8)
        self._arrived_count = int(self._have_arrived.sum())

    def step(self):
        n = self._number_of_nodes

        # -------- Phase 1: collect until all workers have contributed once
        last_finisher = None
        while self._arrived_count < n:
            available_time, i, _ = heapq.heappop(self._heap)
            self._time = available_time

            g = self._asarray_like_point(self._transport.call_ready_node(self._time, i))
            first = (self._have_arrived[i] == 0)
            self._inc_mean(self._mean_curr, self._cnt_curr, i, g, n, self._global_mean)
            if first:
                self._have_arrived[i] = 1
                self._arrived_count += 1

            if self._arrived_count == n:
                # i is the LAST finisher → do NOT reschedule at OLD point
                last_finisher = i
                break
            else:
                # keep non-last workers busy at the old point
                tnext = self._transport.call_available_node_method(
                    self._time, i, node_method="calculate_stochastic_gradient", point=self._point
                )
                heapq.heappush(self._heap, (tnext, i, self._iter))

        # -------- Phase 2: exactly n updates (start with last finisher)
        # First update with CURRENT global mean
        self._point -= self._gamma * self._global_mean
        self._iter += 1

        # Reschedule last finisher with the NEW point
        tnext = self._transport.call_available_node_method(
            self._time, last_finisher, node_method="calculate_stochastic_gradient", point=self._point
        )
        heapq.heappush(self._heap, (tnext, last_finisher, self._iter))

        # Everyone except last finisher still needs one update
        needs_update = np.ones(n, dtype=np.int8)
        needs_update[last_finisher] = 0
        remaining = n - 1

        while remaining > 0:
            available_time, i, _ = heapq.heappop(self._heap)
            self._time = available_time

            g = self._asarray_like_point(self._transport.call_ready_node(self._time, i))

            if needs_update[i]:
                # Fold old-point grad into CURRENT table, update once, then send new point
                self._inc_mean(self._mean_curr, self._cnt_curr, i, g, n, self._global_mean)
                self._point -= self._gamma * self._global_mean
                self._iter += 1
                needs_update[i] = 0
                remaining -= 1

                tnext = self._transport.call_available_node_method(
                    self._time, i, node_method="calculate_stochastic_gradient", point=self._point
                )
                heapq.heappush(self._heap, (tnext, i, self._iter))
            else:
                # Already updated this cycle → stash into NEXT (new-point grads)
                self._inc_mean(self._mean_next, self._cnt_next, i, g, n, self._global_mean_next)
                tnext = self._transport.call_available_node_method(
                    self._time, i, node_method="calculate_stochastic_gradient", point=self._point
                )
                heapq.heappush(self._heap, (tnext, i, self._iter))

        # -------- Prepare next cycle
        self._finish_cycle_and_prepare_next()

    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(
            node_method='calculate_function', point=self._point))

    def get_point(self):
        return self._point

    def get_time(self):
        return self._time

# @FactoryAsyncMaster.register("Ringleader")
# class RingleaderASGD(object):
#     def __init__(self, transport, point, gamma=None, gamma_multiply=None, meta=None):
#         self._transport = transport
#         self._transport.reset_all_nodes(0)
#         self._point = point
#         if gamma_multiply is not None:
#             gamma *= gamma_multiply
#         self._gamma = gamma
#         self._time = 0
        
#         self._heap = []
#         self._iter = 0
#         self._number_of_nodes = self._transport.get_number_of_nodes()
        
#         for node_index in range(self._transport.get_number_of_nodes()):
#             available_time = self._transport.call_available_node_method(
#                 self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
#             heapq.heappush(self._heap, (available_time, node_index, self._iter))

#         n = self._number_of_nodes
#         self._gradients = [np.zeros_like(self._point) for _ in range(n)]
#         self._gradient_counts = [0] * n
#         self._nodes_arrived = [False] * n

#     def step(self):
#         n = self._number_of_nodes
#         while sum(self._nodes_arrived) < n:
#             available_time, node_index, _ = heapq.heappop(self._heap)
#             self._time = available_time

#             g = self._transport.call_ready_node(self._time, node_index)
#             g = np.asarray(g, dtype=self._point.dtype)

#             if self._gradient_counts[node_index] == 0:
#                 self._nodes_arrived[node_index] = True
#             self._gradient_counts[node_index] += 1
#             np.add(self._gradients[node_index], g, out=self._gradients[node_index])

#             if sum(self._nodes_arrived) == n:
#                 break

#             next_time = self._transport.call_available_node_method(
#                 self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
#             heapq.heappush(self._heap, (next_time, node_index, self._iter))

#         self.step_with_node(node_index=node_index) # do one step with the last client that finished previous loop
#         self._nodes_arrived[node_index] = False

#         temp_gradients = [np.zeros_like(self._point) for _ in range(n)]
#         temp_gradient_counts = [0] * n
#         temp_nodes_arrived = [False] * n
#         while sum(self._nodes_arrived) > 0:
#             available_time, node_index, _ = heapq.heappop(self._heap)
#             self._time = available_time

#             if self._nodes_arrived[node_index]:
#                 g = self._transport.call_ready_node(self._time, node_index)
#                 g = np.asarray(g, dtype=self._point.dtype)
#                 self._gradient_counts[node_index] += 1
#                 np.add(self._gradients[node_index], g, out=self._gradients[node_index])
#                 self._nodes_arrived[node_index] = False
#                 self.step_with_node(node_index=node_index)
#             else:
#                 g = self._transport.call_ready_node(self._time, node_index)
#                 g = np.asarray(g, dtype=self._point.dtype)
#                 temp_gradient_counts[node_index] += 1
#                 np.add(temp_gradients[node_index], g, out=temp_gradients[node_index])
#                 temp_nodes_arrived[node_index] = True

#                 next_time = self._transport.call_available_node_method(
#                             self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
#                 heapq.heappush(self._heap, (next_time, node_index, self._iter))

#         self._gradients = temp_gradients
#         self._gradient_counts = temp_gradient_counts
#         self._nodes_arrived = temp_nodes_arrived

#     def step_with_node(self, node_index):
#         G = np.stack(self._gradients, axis=0)
#         counts = np.asarray(self._gradient_counts, dtype=float)
#         per_node_means = G / counts[:, None]
#         global_grad = per_node_means.mean(axis=0)

#         self._point = self._point - self._gamma * global_grad
#         self._iter += 1

#         next_time = self._transport.call_available_node_method(
#                 self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
#         heapq.heappush(self._heap, (next_time, node_index, self._iter))

#     def calculate_function(self):
#         return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
#                                                          point=self._point))
        
#     def get_point(self):
#         return self._point
    
#     def get_time(self):
#         return self._time

@FactoryAsyncMaster.register("MaleniaSGD")
class MaleniaSGD(object):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, seed=None, meta=None):
        self._transport = transport
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._seed = seed
        self._time = 0
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._total_worker_time = 0
    
    def find_all_available_times(self):
        self._transport.reset_all_nodes(0)
        self._heap = [(self._transport.call_available_node_method(self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point),
                        node_index, self._iter)
                            for node_index in range(self._number_of_nodes)]
        heapq.heapify(self._heap) 

    def step(self):
        self.find_all_available_times()
        n = self._number_of_nodes

        start_time = self._time
        gradients = [np.zeros_like(self._point) for _ in range(n)]
        gradient_counts = [0] * n
        nodes_arrived = 0
        while nodes_arrived < n:
            available_time, node_index, _ = heapq.heappop(self._heap)
            self._time = available_time
            stochastic_gradient = self._transport.call_ready_node(self._time, node_index)

            if gradient_counts[node_index] == 0:
                nodes_arrived += 1
            gradient_counts[node_index] += 1
            np.add(gradients[node_index], stochastic_gradient, out=gradients[node_index])

            next_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            heapq.heappush(self._heap, (next_time, node_index, self._iter))

        self._total_worker_time += (self._time - start_time) * n

        G = np.stack(gradients, axis=0)
        counts = np.asarray(gradient_counts, dtype=float)
        per_node_means = G / counts[:, None]
        global_grad = per_node_means.mean(axis=0)

        self._point = self._point - self._gamma * global_grad
        self._iter += 1

 
    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
        
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time
    
    def get_total_worker_time(self):
        return self._total_worker_time


@FactoryAsyncMaster.register("rennala_master")
class RennalaSGD(object):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, batch_size=None, seed=None, meta=None):
        self._transport = transport
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._batch_size = batch_size
        self._seed = seed
        self._time = 0
        
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        
        self._gradient_estimator = 0
        self._current_batch = 0
        self._total_worker_time = 0
    
    def find_all_available_times(self):
        self._transport.reset_all_nodes(0)
        self._heap = [(self._transport.call_available_node_method(self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point),
                        node_index, self._iter)
                            for node_index in range(self._number_of_nodes)]
        if self._batch_size < self._number_of_nodes:
            self._heap = heapq.nsmallest(self._batch_size, self._heap)
        heapq.heapify(self._heap) 

    def step(self):
        self.find_all_available_times()
        
        start_time = self._time
        while self._current_batch < self._batch_size:
            available_time, node_index, _ = heapq.heappop(self._heap)
            self._time = available_time
            stochastic_gradient = self._transport.call_ready_node(self._time, node_index)

            if available_time == np.inf:
                return
            
            self._gradient_estimator = self._gradient_estimator + stochastic_gradient
            self._current_batch += 1

            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            heapq.heappush(self._heap, (available_time, node_index, self._iter))

        self._total_worker_time += (self._time - start_time) * self._number_of_nodes

        self._point = self._point - self._gamma * (self._gradient_estimator / self._current_batch)
        self._iter += 1
        self._current_batch = 0
        self._gradient_estimator = 0

 
    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
        
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time
    
    def get_total_worker_time(self):
        return self._total_worker_time

@FactoryAsyncMaster.register("rennala_fixed_batch")
class RennalaWithFixedBatchSizes(object):
    def __init__(self, transport, point, number_of_gradients, gamma=None, gamma_multiply=None, batch_size=None, seed=None, meta=None):
        self._transport = transport
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._batch_size = batch_size
        self._seed = seed
        self._time = 0
        
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._number_of_gradients = number_of_gradients

        self._gradient_estimator = 0
        self._current_batch = 0
        self._total_worker_time = 0
    
    def step(self):
        if self._time == 0:
            self._transport.reset_all_nodes(0)
            
        time = [self._time] * self._number_of_nodes
        for node_index in range(self._number_of_nodes):
            self._transport._current_time = time[node_index]
            for _ in range(self._number_of_gradients[node_index]):
                available_time = self._transport.call_available_node_method(
                    time[node_index], node_index, node_method="calculate_stochastic_gradient", point=self._point)
                if available_time == np.inf:
                    continue
                self._total_worker_time += available_time - time[node_index]
                time[node_index] = available_time

                stochastic_gradient = self._transport.call_ready_node(time[node_index], node_index)
                self._gradient_estimator = self._gradient_estimator + stochastic_gradient
                self._current_batch += 1

        self._time = max(time)

        self._point = self._point - self._gamma * (self._gradient_estimator / self._current_batch)
        self._iter += 1
        self._current_batch = 0
        self._gradient_estimator = 0
        self._transport.reset_all_nodes(self._time)
 
    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
        
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time
    
    def get_total_worker_time(self):
        return self._total_worker_time


@FactoryAsyncMaster.register("rennala_muon")
class RennalaMuonSGD(object):
    def __init__(
        self,
        transport,
        point,
        gamma=None,
        gamma_multiply=None,
        batch_size=None,
        beta=0.95,
        ns_steps=5,
        nesterov=True,
        seed=None,
        meta=None,
    ):
        self._transport = transport
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._batch_size = batch_size
        self._beta = beta
        self._ns_steps = ns_steps
        self._nesterov = nesterov
        self._seed = seed
        self._time = 0

        self._momentum = np.zeros_like(self._point, dtype=np.float64)
        self._parameter_infos = _build_parameter_infos(meta, self._point.size)
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()

        self._gradient_estimator = 0
        self._current_batch = 0
        self._total_worker_time = 0

    def find_all_available_times(self):
        self._transport.reset_all_nodes(0)
        self._heap = [(self._transport.call_available_node_method(self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point),
                        node_index, self._iter)
                            for node_index in range(self._number_of_nodes)]
        if self._batch_size < self._number_of_nodes:
            self._heap = heapq.nsmallest(self._batch_size, self._heap)
        heapq.heapify(self._heap)

    def step(self):
        self.find_all_available_times()

        start_time = self._time
        while self._current_batch < self._batch_size:
            available_time, node_index, _ = heapq.heappop(self._heap)
            self._time = available_time
            stochastic_gradient = self._transport.call_ready_node(self._time, node_index)

            if available_time == np.inf:
                return

            self._gradient_estimator = self._gradient_estimator + stochastic_gradient
            self._current_batch += 1

            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            heapq.heappush(self._heap, (available_time, node_index, self._iter))

        self._total_worker_time += (self._time - start_time) * self._number_of_nodes

        average_gradient = self._gradient_estimator / self._current_batch
        if self._parameter_infos is None:
            muon_update, self._momentum = _muon_update_numpy(
                average_gradient,
                self._momentum,
                beta=self._beta,
                ns_steps=self._ns_steps,
                nesterov=self._nesterov,
            )
        else:
            muon_update, self._momentum = _structured_muon_update_numpy(
                average_gradient,
                self._momentum,
                self._parameter_infos,
                beta=self._beta,
                ns_steps=self._ns_steps,
                nesterov=self._nesterov,
            )
        self._point = self._point - self._gamma * muon_update
        self._iter += 1
        self._current_batch = 0
        self._gradient_estimator = 0

    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))

    def get_point(self):
        return self._point

    def get_time(self):
        return self._time

    def get_total_worker_time(self):
        return self._total_worker_time


def probabilistic_round(x):
    return int(np.floor(x) + np.random.binomial(size=1, n=1, p=x-np.floor(x)))

@FactoryAsyncMaster.register("rennala_EG_RR")
class Rennala_EG_RR(object):
    def __init__(self, transport, point, eg_stepsize, gamma=None, gamma_multiply=None, batch_size=None, seed=None, meta=None):
        self._transport = transport
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._eg_stepsize = eg_stepsize
        self._batch_size = batch_size
        self._seed = seed
        self._time = 0
        
        
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()

        self._gradient_estimator = 0
        self._current_batch = 0
        self._distribution = np.array([1/self._number_of_nodes] * self._number_of_nodes)
    
    def step(self):
        if self._time == 0:
            self._transport.reset_all_nodes(0)

        vfunc = np.vectorize(probabilistic_round)
        self._number_of_gradients = vfunc(self._distribution * self._batch_size)
        # if self._iter/50:
        #     print('Rennala_EG_RR Allocation: ', self._number_of_gradients)
        #     print("Number of Gradients: ", np.linalg.norm(self._number_of_gradients, 1))

        time = [self._time] * self._transport.get_number_of_nodes()
        for node_index in range(self._transport.get_number_of_nodes()):
            self._transport._current_time = time[node_index]
            for _ in range(self._number_of_gradients[node_index]):
                available_time = self._transport.call_available_node_method(
                    time[node_index], node_index, node_method="calculate_stochastic_gradient", point=self._point)
                if available_time == np.inf:
                    continue
                time[node_index] = available_time

                stochastic_gradient = self._transport.call_ready_node(time[node_index], node_index)
                self._gradient_estimator = self._gradient_estimator + stochastic_gradient
                self._current_batch += 1

        current_iteration_time = np.array(time) - self._time
        self._number_of_gradients[np.where(self._number_of_gradients == 0)] = 1
        current_iteration_time /= self._number_of_gradients

        self._time = max(time)
        self._point = self._point - self._gamma * (self._gradient_estimator / self._current_batch)
        self._iter += 1
        self._current_batch = 0
        self._gradient_estimator = 0
        self._transport.reset_all_nodes(self._time)

        index_to_update = np.argmax(current_iteration_time)
        self._distribution[index_to_update] *= np.exp(-self._eg_stepsize*current_iteration_time[index_to_update])
        self._distribution /= np.linalg.norm(self._distribution, 1)

    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
        
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time


def find_optimal_number_of_gradients_using_harmonic_mean(self, times, batch_size):
    A = np.column_stack((np.arange(self._number_of_nodes), times))
    A = A[A[:, 1].argsort()]

    # Loop to calculate t'(j) for each j and find the minimum
    min_value = float('inf')
    min_index = -1
    for j in range(1, len(times) + 1):
        sum_reciprocal = np.sum(1 / A[:j, 1])
        t_prime_j = (1 / sum_reciprocal) * (batch_size + j)
        
        # Check if this is the minimum value so far
        if t_prime_j < min_value:
            min_value = t_prime_j
            min_index = j

    B = np.zeros_like(times)
    for j,index in enumerate(A[:min_index, 0].astype("int")):
        B[index] = np.ceil(min_value / A[j, 1] - 1)
    B = B.astype(int)

    return B
    
def find_optimal_number_of_gradients(scores, batch_size):
    length = len(scores)
    
    if batch_size == 1:
        if np.any(scores <= 0):
            best_allocation_index = np.random.choice(np.where(scores <= 0)[0], 1)
        else:
            best_allocation_index = np.random.choice(np.where(scores == np.min(scores))[0], 1)[0]
        return np.reshape(np.eye(length, dtype='int')[best_allocation_index], (length, ))
    
    best_prev_allocation = find_optimal_number_of_gradients(scores=scores, batch_size=batch_size-1)
    
    # If any of the negative scores are not included - include one of them
    neg_indices = np.where(scores <= 0)[0]
    if np.any(best_prev_allocation[neg_indices] == 0):
        # Choose a random index where best_prev_allocation is 0
        zero_index = neg_indices[np.random.choice(np.where(best_prev_allocation[neg_indices] == 0)[0], 1)]
        # Set that index to 1
        best_prev_allocation[zero_index] = 1
        return best_prev_allocation
        
    # Mask where scores are not -np.inf
    valid_indices = scores > 0
    # Create an array of allocations
    allocations = np.tile(best_prev_allocation, (length, 1)) + np.eye(length, dtype=int)
    # Compute the corresponding times (max of scores * allocation) for valid indices
    times = np.max(scores * allocations, axis=0)
    # Set the times for invalid indices (where scores are -np.inf) to a large value
    times[~valid_indices] = np.inf
    # Find the index of the minimum time
    best_allocation_index = np.argmin(times)
        
    return allocations[best_allocation_index]

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

    zero_indices = np.where(scores == 0)[0]
    number_of_zero_indices = len(zero_indices)
    if number_of_zero_indices:
        allocation = np.zeros(length, dtype='int')
        allocation[zero_indices] += batch_size//number_of_zero_indices
        indices = np.random.choice(zero_indices, batch_size%number_of_zero_indices, replace=False)
        allocation[indices] += 1
        return allocation

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

def discordia(scores, batch_size, negative_strategy=None):
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
        return allocation + discordia(scores=scores, batch_size=batch_size-number_of_inf_indices, negative_strategy=negative_strategy)
    
    neg_scores = scores < 0
    scores[~neg_scores] = np.inf
    if negative_strategy == "uniform":
        scores[neg_scores] = 0
    elif negative_strategy == "abs":
        scores[neg_scores] = np.abs(scores[neg_scores])
    elif negative_strategy == "lift":
        scores[neg_scores] -= 2*np.min(scores)
    return harmonia(scores=scores, batch_size=batch_size)

@FactoryAsyncMaster.register("ATA")
class ATA(object):
    def __init__(self, transport, point, gamma=None, allocation_type='adaptive', negative_strategy=None, number_of_gradients=None, alpha=None, print_aloc=False, time_means=None, gamma_multiply=None, batch_size=None, seed=None, meta=None):
        self._transport = transport
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._batch_size = batch_size
        self._print_aloc = print_aloc
        self._allocation_type = allocation_type
        self._time_means = time_means # We use this only for printing the proxi regret
        self._alpha = alpha
        self._seed = seed
        self._time = 0
        self._negative_strategy = negative_strategy
        if self._allocation_type == "fixed":
            self._number_of_gradients = number_of_gradients
        
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._total_number_of_gradients = np.array([0] * self._transport.get_number_of_nodes())
        self._mean_estimators = np.array([0.0] * self._transport.get_number_of_nodes())
        self._second_moment_estimator = np.array([0.0] * self._transport.get_number_of_nodes())
        self._variance_estimators = np.array([0.0] * self._transport.get_number_of_nodes())
        self._confidence_score = np.array([-np.inf] * self._number_of_nodes)
        self._gradient_estimator = 0.0
        self._current_batch = 0
        self._total_worker_time = 0
        self._warm_start = np.infty

    def allocate(self):
        if self._allocation_type == "uniform":
            self._number_of_gradients = np.zeros(self._number_of_nodes, dtype=int)
            self._number_of_gradients += self._batch_size//self._number_of_nodes
            indices = np.random.choice(self._number_of_nodes, self._batch_size%self._number_of_nodes, replace=False)
            self._number_of_gradients[indices] += 1
        elif self._allocation_type == "adaptive":
            scores = np.copy(self._confidence_score)
            if np.any(scores <=0):
                self._number_of_gradients = discordia(scores=scores, batch_size=self._batch_size, negative_strategy=self._negative_strategy)
            else:
                self._number_of_gradients = harmonia(scores=scores, batch_size=self._batch_size)

    
    def step(self):
        if self._time == 0:
            self._transport.reset_all_nodes(0)

        self._iter += 1
        self.allocate()
        assert self._number_of_gradients.sum() == self._batch_size

        # if self._print_aloc and self._iter < 100:
        if self._print_aloc and (self._iter % 1000 == 0):
            print(self._confidence_score)
            print("Allocation:", self._number_of_gradients)

            # import matplotlib.pyplot as plt
            # import matplotlib.animation as animation

            # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # def update_plot(frame):
            #     ax1.clear()
            #     ax2.clear()
                
            #     ax1.bar(range(len(self._number_of_gradients)), self._number_of_gradients)
            #     ax1.set_title('Allocation')
            #     ax1.set_xlabel('Node Index')
            #     ax1.set_ylabel('Number of Gradients')
                
            #     ax2.bar(range(len(self._confidence_score)), self._confidence_score)
            #     ax2.set_title('Confidence Scores')
            #     ax2.set_xlabel('Node Index')
            #     ax2.set_ylabel('Confidence Score')

            # ani = animation.FuncAnimation(fig, update_plot, frames=range(100), repeat=False)
            # ani.save('allocation_and_confidence_scores.gif', writer='imagemagick')
            # plt.close(fig)

        time = [self._time] * self._number_of_nodes
        for node_index in range(self._number_of_nodes):
            self._transport._current_time = time[node_index]
            for _ in range(self._number_of_gradients[node_index]):
                available_time = self._transport.call_available_node_method(
                    time[node_index], node_index, node_method="calculate_stochastic_gradient", point=self._point)

                single_iteration_time = available_time - time[node_index]
                # Update total worker time
                self._total_worker_time += single_iteration_time
                # Update mean estimator
                self._mean_estimators[node_index] = (self._total_number_of_gradients[node_index] * self._mean_estimators[node_index] + single_iteration_time) / (self._total_number_of_gradients[node_index] + 1)
                # Update second moment estimator
                self._second_moment_estimator[node_index] = (self._total_number_of_gradients[node_index] * self._second_moment_estimator[node_index] + single_iteration_time**2) / (self._total_number_of_gradients[node_index] + 1)
                # Update the number of samples
                self._total_number_of_gradients[node_index] += 1
                # Update the total time for the worker
                time[node_index] = available_time

                stochastic_gradient = self._transport.call_ready_node(time[node_index], node_index)
                self._gradient_estimator = self._gradient_estimator + stochastic_gradient
                self._current_batch += 1

            # Calculate variance estimator    
            self._variance_estimators[node_index] = self._second_moment_estimator[node_index] - self._mean_estimators[node_index]**2

        self._time = max(time)
        self._point = self._point - self._gamma * (self._gradient_estimator / self._current_batch)
        # update the point in an optimizer style
        self._current_batch = 0
        self._gradient_estimator = 0
        self._transport.reset_all_nodes(self._time)

        if self._allocation_type in ['fixed', 'uniform']:
            return
        
        # Update confidence score - It is important to update after so that you update for all of them!
        valid_indices = self._total_number_of_gradients != 0
        if self._alpha == 'empirical':
            self._confidence_score[valid_indices] = (
                self._mean_estimators[valid_indices]
                - 4 * np.e * 2 * self._mean_estimators[valid_indices] *                     
                    (
                        np.sqrt(np.log(2*self._iter**2) / self._total_number_of_gradients[valid_indices])
                        +       np.log(2*self._iter**2) / self._total_number_of_gradients[valid_indices]
                    )
            )
        else:
            self._confidence_score[valid_indices] = (
                self._mean_estimators[valid_indices]
                - 4 * np.e * self._alpha * 
                    (
                        np.sqrt(np.log(2*self._iter**2) / self._total_number_of_gradients[valid_indices])
                        +       np.log(2*self._iter**2) / self._total_number_of_gradients[valid_indices]
                    )   
            )

        if np.all(self._confidence_score > 0) and (self._warm_start == np.inf) and (self._iter > 1):
            self._warm_start = (self._iter, self._time)

    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time
    
    def get_mean_time(self):
        return np.max(self._number_of_gradients * self._time_means)

    def get_total_worker_time(self):
        return self._total_worker_time
    
    def get_allocation(self):
        return self._number_of_gradients
    
    def get_warm_start(self):
        return self._warm_start


class RennalaSGD_Bandit(object):
    def __init__(self, time_sampler, number_of_nodes, batch_size=None, seed=None):
        self._time_sampler = time_sampler
        self._batch_size = batch_size
        self._seed = seed
        self._time = 0
        self._iter = 0
        self._number_of_nodes = number_of_nodes
        self._total_worker_time = 0
        self._heap = []
    
    def find_all_available_times(self):
        self._heap = [(self._time_sampler(node_index), node_index) for node_index in range(self._number_of_nodes)]
        if self._batch_size < self._number_of_nodes:
            self._heap = heapq.nsmallest(self._batch_size, self._heap)
        heapq.heapify(self._heap) 

    def step(self):
        self.find_all_available_times()
        current_batch = 0
        iteration_time = 0
        while current_batch < self._batch_size:
            time, node_index = heapq.heappop(self._heap)
            iteration_time = time
            current_batch += 1
            heapq.heappush(self._heap, (iteration_time+self._time_sampler(node_index), node_index))

        self._total_worker_time += iteration_time * self._number_of_nodes
        self._time += iteration_time
        self._iter += 1
 
    def get_time(self):
        return self._time
    
    def get_total_worker_time(self):
        return self._total_worker_time
    
class ATA_bandit(object):
    def __init__(self, time_sampler, number_of_nodes, batch_size=None, allocation_type='adaptive', alpha=None, negative_strategy=None,
                  number_of_gradients=None, print_aloc=False, time_means=None, seed=None):
        self._time_sampler = time_sampler
        self._batch_size = batch_size
        self._print_aloc = print_aloc
        self._allocation_type = allocation_type
        self._time_means = time_means # We use this only for printing the proxi regret
        self._alpha = alpha
        self._seed = seed
        self._time = 0
        self._negative_strategy = negative_strategy
        if self._allocation_type == "fixed":
            self._number_of_gradients = number_of_gradients
        
        self._iter = 0
        self._number_of_nodes = number_of_nodes
        self._total_number_of_gradients = np.array([0] * self._number_of_nodes)
        self._mean_estimators = np.array([0.0] * self._number_of_nodes)
        self._second_moment_estimator = np.array([0.0] * self._number_of_nodes)
        self._confidence_score = np.array([-np.inf] * self._number_of_nodes)

        self._total_worker_time = 0
        self._warm_start = np.infty

    def allocate(self):
        if self._allocation_type == "uniform":
            self._number_of_gradients = np.zeros(self._number_of_nodes, dtype=int)
            self._number_of_gradients += self._batch_size//self._number_of_nodes
            indices = np.random.choice(self._number_of_nodes, self._batch_size%self._number_of_nodes, replace=False)
            self._number_of_gradients[indices] += 1
        elif self._allocation_type == "adaptive":
            scores = np.copy(self._confidence_score)
            if np.any(scores <=0):
                self._number_of_gradients = discordia(scores=scores, batch_size=self._batch_size, negative_strategy=self._negative_strategy)
            else:
                self._number_of_gradients = harmonia(scores=scores, batch_size=self._batch_size)

    
    def step(self):
        self._iter += 1
        self.allocate()
        assert self._number_of_gradients.sum() == self._batch_size

        if self._print_aloc and (self._iter % 1000 == 0):
            print(self._confidence_score)
            print("Allocation:", self._number_of_gradients)

        times = [
            sum(self._time_sampler(i) for _ in range(self._number_of_gradients[i]))
            for i in range(self._number_of_nodes)
        ]
        # Update wall clock time
        self._time += max(times)
        # Update total worker time
        self._total_worker_time += sum(times)

        times = np.array(times)
        active_nodes = self._number_of_gradients > 0
        # Update mean estimator
        self._mean_estimators[active_nodes] = (self._total_number_of_gradients[active_nodes] * self._mean_estimators[active_nodes] + times[active_nodes]) / (self._total_number_of_gradients[active_nodes] + self._number_of_gradients[active_nodes])
        # Update the number of samples
        self._total_number_of_gradients[active_nodes] += self._number_of_gradients[active_nodes]

        if self._allocation_type in ['fixed', 'uniform']:
            return
        
        # Update confidence score - It is important to update after so that you update for all of them!
        valid_indices = self._total_number_of_gradients != 0
        if self._alpha == 'empirical':
            self._confidence_score[valid_indices] = (
                self._mean_estimators[valid_indices]
                - 4 * np.e * 2 * self._mean_estimators[valid_indices] *                     
                    (
                        np.sqrt(np.log(2*self._iter**2) / self._total_number_of_gradients[valid_indices])
                        +       np.log(2*self._iter**2) / self._total_number_of_gradients[valid_indices]
                    )
            )
        else:
            self._confidence_score[valid_indices] = (
                self._mean_estimators[valid_indices]
                - 4 * np.e * self._alpha * 
                    (
                        np.sqrt(np.log(2*self._iter**2) / self._total_number_of_gradients[valid_indices])
                        +       np.log(2*self._iter**2) / self._total_number_of_gradients[valid_indices]
                    )   
            )

        if np.all(self._confidence_score > 0) and (self._warm_start == np.inf) and (self._iter > 1):
            self._warm_start = (self._iter, self._time)

    def get_time(self):
        return self._time
    
    def get_mean_time(self):
        return np.max(self._number_of_gradients * self._time_means)

    def get_total_worker_time(self):
        return self._total_worker_time
    
    def get_allocation(self):
        return self._number_of_gradients
    
    def get_warm_start(self):
        return self._warm_start
    
@FactoryAsyncMaster.register("MindFlayer_UCB")
class MindFlayerUCB(object):
    def __init__(self, transport, point, alpha=2, gamma=None, gamma_multiply=None, batch_size=None, seed=None, meta=None):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._alpha = alpha
        self._batch_size = batch_size
        self._seed = seed
        self._time = 0
        
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._total_number_of_gradients = np.array([0] * self._transport.get_number_of_nodes())
        self._mean_estimators = np.array([0] * self._transport.get_number_of_nodes())
        self._confidence_score = np.array([-np.inf] * self._number_of_nodes)
        self._gradient_estimator = 0
        self._current_batch = 0
    
    def step(self):
        self._iter += 1
        self._number_of_gradients = find_optimal_number_of_gradients(self._confidence_score, self._batch_size)
        if self._iter/1000:
            print("RennalaUCB Allocation: ", self._number_of_gradients)

        # if self._iter == 1:
        #     self._number_of_gradients = [1] * self._number_of_nodes

        time = [self._time] * self._number_of_nodes
        for node_index in range(self._transport.get_number_of_nodes()):
            self._transport._current_time = time[node_index]
            for _ in range(self._number_of_gradients[node_index]):
                available_time = self._transport.call_available_node_method(
                    time[node_index], node_index, node_method="calculate_stochastic_gradient", point=self._point)
                if available_time == np.inf:
                    continue
                time[node_index] = available_time

                stochastic_gradient = self._transport.call_ready_node(time[node_index], node_index)
                self._gradient_estimator = self._gradient_estimator + stochastic_gradient
                self._current_batch += 1

            if self._number_of_gradients[node_index] == 0:
                continue
            # Update mean estimator
            self._mean_estimators[node_index] = (self._total_number_of_gradients[node_index] * self._mean_estimators[node_index] + time[node_index] - self._time) / (self._total_number_of_gradients[node_index] + self._number_of_gradients[node_index])
            # Update the number of samples
            self._total_number_of_gradients[node_index] += self._number_of_gradients[node_index]

        # Update confidence score - It is important to update after so that you update for all of them!
        valid_indices = self._total_number_of_gradients != 0
        self._confidence_score[valid_indices] = (
            self._mean_estimators[valid_indices]
            - np.sqrt(2 * self._alpha * np.log(self._iter) / self._total_number_of_gradients[valid_indices])
        )

        self._time = max(time)
        self._point = self._point - self._gamma * (self._gradient_estimator / self._current_batch)
        self._current_batch = 0
        self._gradient_estimator = 0
        self._transport.reset_all_nodes(self._time)

    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time
    
@FactoryAsyncMaster.register("bandit_rennala")
class RennalaWithBandits(object):
    def __init__(self, transport, point, sigma, eta, gamma=None, gamma_multiply=None, batch_size=None, seed=None, meta=None):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._eta = eta
        self._sigma = sigma
        self._batch_size = batch_size
        self._seed = seed
        self._time = 0
        
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._total_number_of_gradients = np.array([0] * self._transport.get_number_of_nodes())
        self._mean_estimators = np.array([0] * self._transport.get_number_of_nodes())
        self._conf_bounds = np.array([np.inf] * self._transport.get_number_of_nodes())

        p = int(np.log2(self._batch_size))
        self._losses = np.array([0] * p)
        self._distribution = np.array([1/p] * p)
        self._losses = [0] * p

        self._gradient_estimator = 0
        self._current_batch = 0

    def set_number_of_gradients(self, support_size):
        scores = np.array(self._mean_estimators - self._conf_bounds)
        A = np.column_stack((np.arange(self._number_of_nodes), scores))
        # sort by their scores
        A = A[A[:, 1].argsort()]

        # assign number of gradients for each workers
        self._number_of_gradients = np.array([0] * self._number_of_nodes)
        self._number_of_gradients[A[:2**support_size, 0].astype("int")] = self._batch_size/(2**support_size)
    
    def step(self):
        if self._time == 0:
            self._transport.reset_all_nodes(0)
    
        p = int(np.log2(self._batch_size))
        support_size = int(np.random.choice(p, 1, p=self._distribution))
        self.set_number_of_gradients(support_size)

        time = [self._time] * self._transport.get_number_of_nodes()
        for node_index in range(self._transport.get_number_of_nodes()):
            self._transport._current_time = time[node_index]
            for _ in range(self._number_of_gradients[node_index]):
                available_time = self._transport.call_available_node_method(
                    time[node_index], node_index, node_method="calculate_stochastic_gradient", point=self._point)
                if available_time == np.inf:
                    continue
                time[node_index] = available_time

                stochastic_gradient = self._transport.call_ready_node(time[node_index], node_index)
                self._gradient_estimator = self._gradient_estimator + stochastic_gradient
                self._current_batch += 1

            if self._number_of_gradients[node_index] + self._total_number_of_gradients[node_index] > 0:
                self._mean_estimators[node_index] = (self._total_number_of_gradients[node_index] * self._mean_estimators[node_index] + time[node_index]) / (self._total_number_of_gradients[node_index] + self._number_of_gradients[node_index])
            self._total_number_of_gradients[node_index] += self._number_of_gradients[node_index]
            self._conf_bounds[node_index] = 2*self._sigma*np.sqrt(np.log(self._iter+1)/ self._total_number_of_gradients[node_index])

        self._losses[support_size] += max(time) - self._time
        self._time = max(time)

        self._point = self._point - self._gamma * (self._gradient_estimator / self._current_batch)
        self._iter += 1
        self._current_batch = 0
        self._gradient_estimator = 0
        self._transport.reset_all_nodes(self._time)

        self._distribution[support_size] = np.exp(-self._eta*self._losses[support_size])
        self._distribution /= np.linalg.norm(self._distribution, ord=1)

        # if self._iter/1000:
        #     print("Uniform Allocation: ", self._number_of_gradients)

    def find_confidence_bounds(self):
        return self._mean_estimators - np.sqrt(2*self._alpha*np.log(self._iter)/ self._total_number_of_gradients)
    
    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
        
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time
    

@FactoryAsyncMaster.register("clipping_rennala_master")
class ClippingRennala(object):
    def __init__(self, transport, point, clipping_times, p, gamma=None, gamma_multiply=None,
                 batch_size=None, seed=None, meta=None):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._clipping_times = clipping_times
        self._p = p
        self._seed = seed
        self._time = 0
        # In this method batch size is the total number of trials
        self._batch_size = batch_size
        
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        
        for node_index in range(self._transport.get_number_of_nodes()):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            
            # For this method we employ a new way to clip clients, instead
            # of pushing into the heap the time the node is ready, we push
            # the minimum of when it is ready with the clipping time
            check_time = min(available_time, self._time + self._clipping_times[node_index])
            heapq.heappush(self._heap, (check_time, node_index, self._iter))
            
        self._gradient_estimator = 0
        self._current_batch = 0
    
    def step(self):
        num_trials = np.zeros(self._number_of_nodes)

        while np.sum(num_trials) < self._batch_size:
            available_time, node_index, iter = heapq.heappop(self._heap)
            self._time = available_time
            num_trials[node_index] += 1

            # the popped value represents a finished calculation
            if self._transport.is_node_available_at_time(self._time, node_index):
                stochastic_gradient = self._transport.call_ready_node(self._time, node_index)

                # with resetting this should always be the case
                assert iter == self._iter
                self._gradient_estimator += stochastic_gradient
                self._current_batch += 1

            else:  # we clipped
                self._transport.ignore_node(self._time, node_index)

            # pushing when over the batch size will be ignored by resetting
            available_time = self._transport.call_available_node_method(
                self._time, node_index,
                node_method="calculate_stochastic_gradient",
                point=self._point)

            check_time = min(available_time, self._time + self._clipping_times[node_index])
            heapq.heappush(self._heap, (check_time, node_index, self._iter))

        #assert self._current_batch == self._batch_size
        estimator = self._gradient_estimator / np.sum(num_trials) * self._p
        self._point = self._point - self._gamma * estimator

        self._iter += 1
        self._current_batch = 0
        self._gradient_estimator = 0

        self._heap = []
        self._transport.reset_all_nodes(self._time)
        for node_idx in range(self._transport.get_number_of_nodes()):
            next_available_time = self._transport.call_available_node_method(
                self._time, node_idx,
                node_method="calculate_stochastic_gradient",
                point=self._point)

            check_time = min(next_available_time, self._time + self._clipping_times[node_idx])
            heapq.heappush(self._heap, (check_time, node_idx, self._iter))

    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
        
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time


@FactoryAsyncMaster.register("dynamic_clipping_rennala_master")
class DynamicClippingRennala(object):
    def __init__(self, transport, point, p, gamma=None, gamma_multiply=None,
                 batch_size=None, seed=None, meta=None):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._p = p
        self._seed = seed
        self._time = 0
        # In this method batch size is the total number of trials
        self._batch_size = batch_size
        
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()

        self._time_samples = [[]] * self._number_of_nodes
        
        for node_index in range(self._transport.get_number_of_nodes()):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            
            # For this method we employ a new way to clip clients, instead
            # of pushing into the heap the time the node is ready, we push
            # the minimum of when it is ready with the clipping time
            clipping_time = self._get_clipping_estimate(node_index)
            if clipping_time:
                check_time = min(available_time, self._time + clipping_time)
            else:
                check_time = available_time
            heapq.heappush(self._heap, (check_time, node_index, self._iter))
            
        self._gradient_estimator = 0
        self._current_batch = 0
    
    def step(self):
        num_trials = np.zeros(self._number_of_nodes)

        while np.sum(num_trials) < self._batch_size:
            available_time, node_index, iter = heapq.heappop(self._heap)
            self._time = available_time
            num_trials[node_index] += 1

            # the popped value represents a finished calculation
            if self._transport.is_node_available_at_time(self._time, node_index):
                stochastic_gradient = self._transport.call_ready_node(self._time, node_index)

                # with resetting this should always be the case
                assert iter == self._iter
                self._gradient_estimator += stochastic_gradient
                self._current_batch += 1

            else:  # we clipped
                self._transport.ignore_node(self._time, node_index)

            # pushing when over the batch size will be ignored by resetting
            available_time = self._transport.call_available_node_method(
                self._time, node_index,
                node_method="calculate_stochastic_gradient",
                point=self._point)

            clipping_time = self._get_clipping_estimate(node_index)
            if clipping_time:
                check_time = min(available_time, self._time + clipping_time)
            else:
                check_time = available_time
            heapq.heappush(self._heap, (check_time, node_index, self._iter))

        #assert self._current_batch == self._batch_size
        estimator = self._gradient_estimator / np.sum(num_trials) * self._p
        self._point = self._point - self._gamma * estimator

        self._iter += 1
        self._current_batch = 0
        self._gradient_estimator = 0

        self._heap = []
        self._transport.reset_all_nodes(self._time)
        for node_idx in range(self._transport.get_number_of_nodes()):
            next_available_time = self._transport.call_available_node_method(
                self._time, node_idx,
                node_method="calculate_stochastic_gradient",
                point=self._point)

            clipping_time = self._get_clipping_estimate(node_index)
            if clipping_time:
                check_time = min(next_available_time, self._time + clipping_time)
            else:
                check_time = next_available_time
            heapq.heappush(self._heap, (check_time, node_idx, self._iter))

    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
        
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time

    def _get_clipping_estimate(self, node_index):
        num_samples = len(self._time_samples[node_index])  
        if num_samples == 0:
            return None

        percentile_index = np.floor((num_samples + 1) * self._p)
        return self._time_samples[node_index][percentile_index]
    
@FactoryAsyncMaster.register("clipping_mindflayer_master")
class ClippingMindFlayer(object):
    def __init__(self, transport, point, T, clipping_time, gamma=None, gamma_multiply=None,
                 batch_size=None, seed=None, meta=None):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._batch_size = batch_size
        self._clipping_time = clipping_time # for now assume a global clipping time
        self._T = T
        self._seed = seed
        self._time = 0
        
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        
        for node_index in range(self._transport.get_number_of_nodes()):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            heapq.heappush(self._heap, (available_time, node_index, self._iter))
            
        self._gradient_estimator = 0
        self._current_batch = 0
    
    def step(self):
        while self._time <= self._T * (self._iter + 1):
            available_time, node_index, iter = heapq.heappop(self._heap)
            time_diff = available_time - self._time
            start_time = self._time

            # collect gradients within a time frame
            while time_diff <= self._clipping_time:
                self._time = available_time
                stochastic_gradient = self._transport.call_ready_node(self._time, node_index)

                if iter == self._iter:
                    self._gradient_estimator += stochastic_gradient
                    self._current_batch += 1

                if len(self._heap) == 0:
                    break

                # loop update
                available_time, node_index, iter = heapq.heappop(self._heap)
                time_diff = available_time - self._time

            self._time = start_time + self._clipping_time
            self._heap = []
            self._transport.reset_all_nodes(self._time)
            for node_idx in range(self._transport.get_number_of_nodes()):
                next_available_time = self._transport.call_available_node_method(
                    self._time, node_idx, node_method="calculate_stochastic_gradient", point=self._point)
                heapq.heappush(self._heap, (next_available_time, node_idx, self._iter))
        
        # Now available_time > Time Frame
        if self._current_batch > 0:
            estimator = self._gradient_estimator / self._current_batch

            #estimator /= (1 - self._p0) # for now we'll assume no p0
            self._point = self._point - self._gamma * estimator

        self._iter += 1
        self._current_batch = 0
        self._gradient_estimator = 0
        self._time = self._T * self._iter

        self._heap = []
        self._transport.reset_all_nodes(self._time)
        for node_idx in range(self._transport.get_number_of_nodes()):
            next_available_time = self._transport.call_available_node_method(
                self._time, node_idx,
                node_method="calculate_stochastic_gradient",
                point=self._point)
            heapq.heappush(self._heap, (next_available_time, node_idx, self._iter))

    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
        
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time


@FactoryAsyncMaster.register("fixed_clipping_master")
class FixedClipping(object):
    def __init__(self, transport, point, num_clips, clipping_times, ps,
                 gamma=None, gamma_multiply=None, batch_size=None, seed=None,
                 meta=None, **kwargs):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._batch_size = batch_size
        self._clipping_times = clipping_times
        self._num_clips = num_clips
        self._ps = ps
        self._seed = seed
        self._time = 0
        
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        
        for node_index in range(self._transport.get_number_of_nodes()):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            
            # For this method we employ a new way to clip clients, instead
            # of pushing into the heap the time the node is ready, we push
            # the minimum of when it is ready with the clipping time
            check_time = min(available_time, self._time + self._clipping_times[node_index])
            heapq.heappush(self._heap, (check_time, node_index, self._iter))
            
            
        self._gradient_estimator = 0
        self._current_batch = 0
    
    def step(self):
        num_trials = np.zeros(self._number_of_nodes)
        
        while (num_trials < self._num_clips).any():
            available_time, node_index, iter = heapq.heappop(self._heap)
            self._time = available_time
            num_trials[node_index] += 1

            # the popped value represents a finished calculation
            if self._transport.is_node_available_at_time(self._time, node_index):
                stochastic_gradient = self._transport.call_ready_node(self._time, node_index)

                # with resetting this should always be the case
                assert iter == self._iter
                self._gradient_estimator += stochastic_gradient
                self._current_batch += 1

            else:  # we clipped
                self._transport.ignore_node(self._time, node_index)

            if num_trials[node_index] < self._num_clips[node_index]:
                available_time = self._transport.call_available_node_method(
                    self._time, node_index,
                    node_method="calculate_stochastic_gradient",
                    point=self._point)

                check_time = min(available_time, self._time + self._clipping_times[node_index])
                heapq.heappush(self._heap, (check_time, node_index, self._iter))

        if self._current_batch > 0:
            estimator = self._gradient_estimator / num_trials.dot(self._ps)

            #estimator /= (1 - self._p0) # for now we'll assume no p0
            self._point = self._point - self._gamma * estimator

        self._iter += 1
        self._current_batch = 0
        self._gradient_estimator = 0

        self._heap = []
        self._transport.reset_all_nodes(self._time)
        for node_idx in range(self._transport.get_number_of_nodes()):
            next_available_time = self._transport.call_available_node_method(
                self._time, node_idx,
                node_method="calculate_stochastic_gradient",
                point=self._point)

            check_time = min(next_available_time, self._time + self._clipping_times[node_idx])
            heapq.heappush(self._heap, (check_time, node_idx, self._iter))

    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
        
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time

@FactoryAsyncMaster.register("mindflayer_clientwise_master")
class MindFlayerClientWise(object):
    def __init__(self, transport, point, gamma, T, p0, gamma_multiply=None,
                 seed=None, meta=None):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._T = T
        self._seed = seed
        self._time = 0

        self._num_nodes = transport.get_number_of_nodes()
        self._dim = point.shape[0]

        self._current_batches = np.zeros(self._num_nodes)
        self._gradient_estimators = np.zeros((self._num_nodes, self._dim))

        self._p0 = p0

        self._heap = []
        self._iter = 0

        # Initialize the heap with the first batch of node computations
        for node_index in range(self._transport.get_number_of_nodes()):
            available_time = self._transport.call_available_node_method(
                self._time, node_index,
                node_method="calculate_stochastic_gradient", point=self._point)
            heapq.heappush(self._heap, (available_time, node_index, self._iter))


    def step(self):
        available_time, node_index, iter = heapq.heappop(self._heap)

        # collect gradients within a time frame
        while available_time <= self._T * (self._iter + 1):
            self._time = available_time
            stochastic_gradient = self._transport.call_ready_node(self._time, node_index)

            if iter == self._iter:
                self._gradient_estimators[node_index] += stochastic_gradient
                self._current_batches[node_index] += 1

            # return popped node into heap
            next_available_time = self._transport.call_available_node_method(
                self._time, node_index,
                node_method="calculate_stochastic_gradient",
                point=self._point)
            heapq.heappush(self._heap, (next_available_time, node_index, self._iter))

            # loop update
            available_time, node_index, iter = heapq.heappop(self._heap)


        # Now available_time > Time Frame
        if np.sum(self._current_batches) > 0:
            estimator = np.zeros(self._dim)
            for i in range(self._num_nodes):
                if self._current_batches[i] != 0:
                    estimator += self._gradient_estimators[i] / self._current_batches[i]

            estimator /= self._num_nodes
            estimator /= (1 - self._p0)
            self._point = self._point - self._gamma * estimator

        self._iter += 1
        self._current_batches = np.zeros(self._num_nodes)
        self._gradient_estimators = np.zeros((self._num_nodes, self._dim))
        self._time = self._T * self._iter

        # Since we are focusing on the case where we do reset, this will remain
        reset = True
        if reset:
            self._heap = []
            self._transport.reset_all_nodes(self._time)
            for node_idx in range(self._transport.get_number_of_nodes()):
                next_available_time = self._transport.call_available_node_method(
                    self._time, node_idx,
                    node_method="calculate_stochastic_gradient",
                    point=self._point)
                heapq.heappush(self._heap, (next_available_time, node_idx, self._iter))
        else: # deprecated but kept
            # remove then add the points which triggered the batch back

            # Notice one point is already popped by while
            removed_nodes = [(available_time, node_index, iter)]

            next_task = heapq.nsmallest(1, self._heap)
            is_next_bigger = next_task[0][0] > available_time

            while not is_next_bigger:
                removed_nodes.append(heapq.heappop(self._heap))

                next_task = heapq.nsmallest(1, self._heap)
                is_next_bigger = next_task[0][0] > available_time if len(next_task) != 0 else True

            for node in removed_nodes:
                available_time, node_index, iter = node
                self._transport.ignore_node(self._time, node_index)

                next_available_time = self._transport.call_available_node_method(
                    self._time, node_index,
                    node_method="calculate_stochastic_gradient",
                    point=self._point)
                heapq.heappush(self._heap, (next_available_time, node_index, self._iter))

            assert(len(self._heap) == self._transport.get_number_of_nodes())


    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
        
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time


@FactoryAsyncMaster.register("mindflayer_master")
class AsynchronousTimeFrameMiniBatchSGD(object):
    def __init__(self, transport, point, gamma, T, p0, gamma_multiply=None,
                 seed=None, meta=None):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._T = T
        self._seed = seed
        self._time = 0

        self._num_nodes = transport.get_number_of_nodes()
        self._dim = point.shape[0]

        self._current_batch = 0
        self._gradient_estimator = 0

        self._p0 = p0

        self._heap = []
        self._iter = 0

        # Initialize the heap with the first batch of node computations
        for node_index in range(self._transport.get_number_of_nodes()):
            available_time = self._transport.call_available_node_method(
                self._time, node_index,
                node_method="calculate_stochastic_gradient", point=self._point)
            heapq.heappush(self._heap, (available_time, node_index, self._iter))


    def step(self):
        available_time, node_index, iter = heapq.heappop(self._heap)

        # collect gradients within a time frame
        while available_time <= self._T * (self._iter + 1):
            self._time = available_time
            stochastic_gradient = self._transport.call_ready_node(self._time, node_index)

            if iter == self._iter:
                self._current_batch += 1
                self._gradient_estimator += stochastic_gradient

            # return popped node into heap
            next_available_time = self._transport.call_available_node_method(
                self._time, node_index,
                node_method="calculate_stochastic_gradient",
                point=self._point)
            heapq.heappush(self._heap, (next_available_time, node_index, self._iter))

            # loop update
            available_time, node_index, iter = heapq.heappop(self._heap)


        # Now available_time > Time Frame
        if self._current_batch > 0:
            estimator = self._gradient_estimator / self._current_batch

            estimator /= (1 - self._p0)
            self._point = self._point - self._gamma * estimator

        self._iter += 1
        self._current_batch = 0
        self._gradient_estimator = 0
        self._time = self._T * self._iter

        # Since we are focusing on the case where we do reset, this will remain
        reset = True
        if reset:
            self._heap = []
            self._transport.reset_all_nodes(self._time)
            for node_idx in range(self._transport.get_number_of_nodes()):
                next_available_time = self._transport.call_available_node_method(
                    self._time, node_idx,
                    node_method="calculate_stochastic_gradient",
                    point=self._point)
                heapq.heappush(self._heap, (next_available_time, node_idx, self._iter))
        else: # deprecated but kept
            # remove then add the points which triggered the batch back

            # Notice one point is already popped by while
            removed_nodes = [(available_time, node_index, iter)]

            next_task = heapq.nsmallest(1, self._heap)
            is_next_bigger = next_task[0][0] > available_time

            while not is_next_bigger:
                removed_nodes.append(heapq.heappop(self._heap))

                next_task = heapq.nsmallest(1, self._heap)
                is_next_bigger = next_task[0][0] > available_time if len(next_task) != 0 else True

            for node in removed_nodes:
                available_time, node_index, iter = node
                self._transport.ignore_node(self._time, node_index)

                next_available_time = self._transport.call_available_node_method(
                    self._time, node_index,
                    node_method="calculate_stochastic_gradient",
                    point=self._point)
                heapq.heappush(self._heap, (next_available_time, node_index, self._iter))

            assert(len(self._heap) == self._transport.get_number_of_nodes())


    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
        
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time


@FactoryAsyncMaster.register("minibatch_sgd_master")
class MiniBatchSGD(object):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, batch_size=None, seed=None, meta=None):
        self._transport = transport
        self._transport.reset_all_nodes(0)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._batch_size = batch_size
        self._seed = seed
        self._time = 0
        
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._current_times = [None for _ in range(self._number_of_nodes)]
        for node_index in range(self._number_of_nodes):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            self._current_times[node_index] = available_time
    
    def step(self):
        max_available_time = -np.inf
        for node_index in range(self._number_of_nodes):
            available_time = self._current_times[node_index]
            max_available_time = max(max_available_time, available_time)
            
        self._time = max_available_time
        gradient_estimator = 0
        for node_index in range(self._number_of_nodes):
            stochastic_gradient = self._transport.call_ready_node(self._time, node_index)
            gradient_estimator = gradient_estimator + stochastic_gradient
        self._point = self._point - self._gamma * (gradient_estimator / self._number_of_nodes)
        for node_index in range(self._number_of_nodes):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            self._current_times[node_index] = available_time
    
    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))
        
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time


def _generate_seed(generator):
    return generator.integers(10e9)


def get_algorithm(functions, point, seed, 
                  algorithm_name, delays, 
                  algorithm_master_params={}, algorithm_node_params={},
                  meta=OptimizationProblemMeta()):
    node_name = algorithm_name + "_node"
    master_name = algorithm_name + "_master"
    node_cls = FactoryAsyncNode.get(node_name)
    master_cls = FactoryAsyncMaster.get(master_name)
    generator = np.random.default_rng(seed)
    nodes = [Signature(node_cls, function, seed=_generate_seed(generator), **algorithm_node_params) 
             for function in functions]
    transport = DelayedAsynchronousTransport(nodes, delays)
    return master_cls(transport, point, seed=seed, meta=meta, **algorithm_master_params)
