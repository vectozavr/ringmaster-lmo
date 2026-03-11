import copy

from enum import Enum


class State(Enum):
    AVAILABLE = 1
    WORKING = 2
    
    
class ReturnState(Enum):
    WAIT = 1
    RESULT = 2

class RandomDelayedAsynchronousTransport(object):
    def __init__(self, nodes, delays, noise_function, is_independent=True):
        self._nodes = [node.create_instance() for node in nodes]
        self._is_independent = is_independent
        self._delays = delays
        self._noise_function = noise_function

        self._states = [State.AVAILABLE] * len(nodes)
        self._current_time = 0
        self._node_call_count = [0] * len(nodes)
        self._outputs = [None] * len(nodes)
        self._time_return = [None] * len(nodes)

        if not self._is_independent:
            # noises are stored by call index to ensure dependency
            self._noise = {}

    def _get_noise_for_node(self, node_index):
        call_index = self._node_call_count[node_index]

        if self._is_independent:
            return self._noise_function(node_index)
        else:
            if call_index not in self._noise:
                self._noise[call_index] = self._noise_function(node_index)
            return self._noise[call_index]
    
    def call_available_node_method(self, time, node_index, node_method, **kwargs):
        assert self._current_time <= time
        self._current_time = time
        assert self._states[node_index] == State.AVAILABLE
        self._states[node_index] = State.WORKING

        noise = self._get_noise_for_node(node_index)
        self._outputs[node_index] = getattr(self._nodes[node_index], node_method)(**kwargs)
        self._time_return[node_index] = self._current_time +\
                self._delays[node_index] + noise
        self._node_call_count[node_index] += 1

        return self._time_return[node_index]
    
    def call_ready_node(self, time, node_index):
        assert self._current_time <= time
        assert self._states[node_index] == State.WORKING
        self._current_time = time
        assert self._current_time >= self._time_return[node_index]
        self._states[node_index] = State.AVAILABLE
        return self._outputs[node_index]

    def ignore_node(self, time, node_index):
        self._current_time = time
        self._states[node_index] = State.AVAILABLE

    def is_node_available_at_time(self, time, node_index):
        return time >= self._time_return[node_index]
    
    def call_nodes_method(self, node_method, **kwargs):
        return [getattr(self._nodes[node_index], node_method)(**kwargs)
                for node_index in range(self.get_number_of_nodes())]
    
    def get_number_of_nodes(self):
        return len(self._nodes)

    def reset_all_nodes(self, time=None):
        self._current_time = time
        self._outputs = [None] * len(self._nodes)
        self._time_return = [None] * len(self._nodes)
        self._states = [State.AVAILABLE] * len(self._nodes)


class DelayedAsynchronousTransport(object):
    def __init__(self, nodes, delays):
        self._nodes = [node.create_instance() for node in nodes]
        self._delays = delays
        self._states = [State.AVAILABLE] * len(nodes)
        self._current_time = 0
        
        self._outputs = [None] * len(nodes)
        self._time_return = [None] * len(nodes)
    
    def call_available_node_method(self, time, node_index, node_method, **kwargs):
        assert self._current_time <= time
        self._current_time = time
        assert self._states[node_index] == State.AVAILABLE
        self._states[node_index] = State.WORKING
        self._outputs[node_index] = getattr(self._nodes[node_index], node_method)(**kwargs)
        self._time_return[node_index] = self._current_time + self._delays[node_index]
        return self._time_return[node_index]
    
    def call_ready_node(self, time, node_index):
        assert self._current_time <= time
        assert self._states[node_index] == State.WORKING
        self._current_time = time
        assert self._current_time >= self._time_return[node_index]
        self._states[node_index] = State.AVAILABLE
        return self._outputs[node_index]
    
    def call_nodes_method(self, node_method, **kwargs):
        return [getattr(self._nodes[node_index], node_method)(**kwargs)
                for node_index in range(self.get_number_of_nodes())]
    
    def get_number_of_nodes(self):
        return len(self._nodes)
    
    def reset_all_nodes(self, time=None):
        self._current_time = time
        self._outputs = [None] * len(self._nodes)
        self._time_return = [None] * len(self._nodes)
        self._states = [State.AVAILABLE] * len(self._nodes)


class MethodDelayedAsynchronousTransport(object):
    def __init__(self, nodes, delays):
        self._nodes = [node.create_instance() for node in nodes]
        self._delays = delays
        self._methods = delays.keys()
        self._states = {k: [State.AVAILABLE] * len(nodes) for k in self._methods}
        self._current_time = 0
        
        self._outputs = {k: [None] * len(nodes) for k in self._methods}
        self._time_return = {k: [None] * len(nodes) for k in self._methods}
    
    def call_available_node_method(self, time, node_index, node_method, **kwargs):
        assert self._current_time <= time
        self._current_time = time
        assert self._states[node_method][node_index] == State.AVAILABLE
        self._states[node_method][node_index] = State.WORKING
        self._outputs[node_method][node_index] = [node_index, node_method, copy.deepcopy(kwargs)]
        cost_method = "cost_" + node_method
        cost = getattr(self._nodes[node_index], cost_method)(**kwargs)
        self._time_return[node_method][node_index] = self._current_time + cost * self._delays[node_method][node_index]
        return self._time_return[node_method][node_index]
    
    def call_ready_node(self, time, node_index, node_method):
        assert self._current_time <= time
        assert self._states[node_method][node_index] == State.WORKING
        self._current_time = time
        assert self._current_time >= self._time_return[node_method][node_index]
        self._states[node_method][node_index] = State.AVAILABLE
        node_index, node_method, kwargs = self._outputs[node_method][node_index]
        return getattr(self._nodes[node_index], node_method)(**kwargs)
    
    def call_node_method(self, node_index, node_method, **kwargs):
        assert node_method not in self._methods
        return getattr(self._nodes[node_index], node_method)(**kwargs)
    
    def call_nodes_method(self, node_method, **kwargs):
        assert node_method not in self._methods
        return [self.call_node_method(node_index, node_method, **kwargs)
                for node_index in range(self.get_number_of_nodes())]
    
    def get_number_of_nodes(self):
        return len(self._nodes)
    
    def get_methods(self):
        return self._methods
    
    def get_delays(self):
        return self._delays
    
    def get_time(self):
        return self._current_time
