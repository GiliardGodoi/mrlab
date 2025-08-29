import random

from collections.abc import Iterable
from itertools import (
    product
)
from scipy.stats import rv_continuous, rv_discrete
from .params import BaseParams

def _check_params_values(search_values):
    for key, value in search_values.items():
        if not isinstance(value, Iterable) or isinstance(value, str):
            search_values[key] = (value,)
    return search_values

def grid_search_params(search_values, initial=None):
    search_values = _check_params_values(search_values)
    keys, values = zip(*search_values.items())
    combos = (dict(zip(keys, v)) for v in product(*values))
    for combo in combos:
        if isinstance(initial, BaseParams) :
            yield initial.update(**combo)
        elif isinstance(initial, dict):
            params = dict(initial)
            params.update(combo)
            yield params
        elif initial is None:
            yield combo
        else:
            raise RuntimeError

class GridSearch:

    def __init__(self, values : dict, initial=None):
        self.search_space = _check_params_values(values)
        self.initial = initial
        keys, values = zip(*self.search_space.items())
        self.combo_iter = (dict(zip(keys, v)) for v in product(*values))

    def __iter__(self):
        return self

    def __next__(self):
        combo = next(self.combo_iter)
        if isinstance(self.initial, BaseParams):
            params = self.initial.update(**combo)
        elif isinstance(self.initial, dict):
            params = dict(self.initial)
            params.update(combo)
        elif self.initial is None :
            params = combo
        else:
            return RuntimeError
        return params

class Sampler:
    def sample(self):
        raise NotImplementedError
    def __str__(self):
        return f"{self.__class__.__name__}: <...>"

class ListSampler(Sampler):
    def __init__(self, values):
        self.values = values

    def sample(self):
        return random.choice(self.values)

class CallableSampler(Sampler):
    def __init__(self, func):
        self.func = func
    def sample(self):
        return self.func()

class ScipySampler(Sampler):
    def __init__(self, dist):
        self.dist = dist

    def sample(self):
        return self.dist.rvs()

class FixedSampler(Sampler):
    def __init__(self, value):
        self.value = value
    def sample(self):
        return self.value

def make_sampler(param):
    if isinstance(param, (rv_continuous, rv_discrete)):
        return ScipySampler(param)
    elif callable(param):
        return CallableSampler(param)
    elif isinstance(param, Iterable) and not isinstance(param, (str, set)):
        return ListSampler(param)
    else:
        return FixedSampler(param)

class RandomSearch:

    def __init__(self, n_samples, values:dict, initial=None):
        self.n_samples = n_samples
        self.initial = initial
        self.samplers = {k : make_sampler(v) for k, v in values.items()}
        self._counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._counter >= self.n_samples:
            raise StopIteration
        combo = {k : s.sample() for k, s in self.samplers.items()}
        if isinstance(self.initial, BaseParams):
            params = self.initial.update(**combo)
        elif isinstance(self.initial, dict):
            params = dict(self.initial)
            params.update(combo)
        elif self.initial is None:
            params = combo
        else:
            raise RuntimeError

        self._counter += 1
        return params

    def reset(self):
        self._counter = 0