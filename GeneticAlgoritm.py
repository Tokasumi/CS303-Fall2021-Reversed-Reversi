import random
import numpy as np
import typing


def _resize_bound(rows, *args):
    returns = []
    for dimension, elem in zip(map(np.ndim, args), args):
        assert 0 <= dimension <= 1
        result_elem = np.tile(elem, (rows, 1)) if dimension == 1 else elem
        returns.append(result_elem)
    return returns


_INIT_RANDOM_FUNCS = {}
_INIT_RANDOM_FUNCS.update(dict.fromkeys(['u', 'uniform'], np.random.uniform))
_INIT_RANDOM_FUNCS.update(dict.fromkeys(['n', 'normal', 'gaussian'], np.random.normal))

_shape_msg = 'Generated shape of arguments does not match the shape initial arguments for population (size, arg_num)'


def initialize(agent_class: typing.ClassVar, size: int = 64, arg_num: int = None, **kwargs):
    """
    3 types of keyword arguments will be accepted: random type, random function or callable object.
    rfunc: function object (or string identifier) that you defined (or choose) to generate initial arguments
    rgen: callable object that you defined (or choose) to generate initial arguments
    rarg: tuple used for rfunc, (low, high) for uniform, (mu, sigma) for gaussian

    difference between rfunc and rgent:
    initial_args = rfunc(*rbias, shape) where shape = (size, arg_num) | initial_args = rgen()
    """
    arg_num = len(agent_class().to_list()) if arg_num is None else arg_num
    arg_shape = (size, arg_num)

    rgen, rfunc = kwargs.get('rgen'), kwargs.get('rfunc', 'gaussian')
    rfunc = _INIT_RANDOM_FUNCS.get(rfunc, rfunc)  # if rfunc is not an valid identifier, treat it as a function
    initial_args = rfunc(*kwargs.get('rarg', (0, 50)), arg_shape) if rgen is None else rgen()
    assert np.shape(initial_args) == arg_shape, _shape_msg

    population, lives = [None] * size, np.zeros(size)
    for i, args_array in enumerate(initial_args):
        agent = agent_class()
        agent.from_list(args_array)
        population[i] = agent
    return population, lives


def sort_eliminate(population: list, scores: typing.Iterable, lives: typing.Iterable = None, ratio=0.5, num=None):
    """
    sort the population by scores, then last (N * ratio) members or last (num) members will be truncated.
    if num is specified, ratio will be ignored, if both num and ratio are not specified, ratio will be 0.5.
    agents still alive will have live + 1
    """
    attributes = zip(population, scores) if lives is None else zip(population, scores, lives)
    attributes = sorted(attributes, key=lambda elem: elem[1], reverse=True)
    # sort zipped object with score, will return a list

    psize = len(population)
    eliminate_index = np.clip(1, psize * (1 - ratio) if num is None else psize - num, psize).astype(int)
    attributes = attributes[:eliminate_index]

    population, scores = [elem[0] for elem in attributes], np.array([elem[1] for elem in attributes])
    lives = None if lives is None else np.array([elem[2] + 1 for elem in attributes])
    return population, scores, lives


def _crossover_selection(population_size, bias=None):
    if bias is None:
        max_idx = population_size - 1
        while True:
            p1, p2 = random.randint(1, max_idx), random.randint(1, max_idx)
            yield (0, p2) if p1 == p2 else (p1, p2)
    else:
        bias = np.array(bias)
        assert bias.ndim == 1 and bias.size == population_size, \
            f'The shape of bias ({bias.shape}) ' \
            f'must be the same as the size of population ({population_size})'
        assert np.all(bias > 0), 'All entry of selection bias must be positive'
        accumulate = np.add.accumulate(bias)
        while True:
            p1 = np.searchsorted(accumulate, random.uniform(0.0, accumulate[-1]), 'left')
            p2_rng = random.uniform(accumulate[p1], accumulate[-1] + accumulate[p1 - 1])
            p2_rng = p2_rng - accumulate[-1] if p2_rng > accumulate[-1] else p2_rng
            p2 = np.searchsorted(accumulate, p2_rng, 'left')
            yield p1, p2


def crossover(population: list,
              lives,
              count: int,
              bounds: typing.Tuple = (-50, 50),
              mutate_rate=0.33,
              mutate_args: typing.Tuple = (0, 10.0),
              mutate_rfunc='gaussian',
              select_bias: typing.Iterable = None
              ):
    """
    Crossover and generate the next generation, the child will pick up parents' arg_list with equal chance.

    :param population: sorted population by any criterion.
    :param lives: lives of member in population.
    :param count: the number of new members to generate.
    :param bounds: lower and upper bound of member arguments, a value or an array.
    :param mutate_rate: mutate rate from 0.0 to 1.0, a value or an array.
    :param mutate_args: mutate mean and standard deviation.
    :param mutate_rfunc: string identifier or a function object, default numpy.random.normal(*mutate_args, args_shape).
    :param select_bias: the chance of selection of parents, if not modified, the chance will be equal.
    :returns: the population with offspring and lives of members
    """
    agent_class = population[0].__class__
    arg_length = len(population[0].to_list())
    args_shape = (count, arg_length)

    choice_matrix = np.random.randint(low=0, high=2, size=args_shape)
    mutate_matrix = np.random.random(size=args_shape) > mutate_rate
    mutate_rfunc = _INIT_RANDOM_FUNCS.get(mutate_rfunc, mutate_rfunc)
    mutate_val = mutate_rfunc(*mutate_args, args_shape)
    mutate_val[mutate_matrix] = 0.0

    offspring = []
    parent_selection = _crossover_selection(len(population), select_bias)
    identity = np.arange(arg_length)
    for i in range(count):
        p1, p2 = next(parent_selection)
        parent_args = np.array((population[p1].to_list(), population[p2].to_list()))
        child_args = parent_args[choice_matrix[i], identity]
        # choice is 0 or 1 array and identity is 0 to len-1
        # arr = [[1,2,3,4,5], [6,7,8,9,0]], choice = [0,1,0,1,0], then arr[choice, identity] = [1,7,3,9,5]
        child_args = np.clip(bounds[0], child_args + mutate_val[i], bounds[1]).squeeze()

        child = agent_class()
        child.from_list(child_args)
        offspring.append(child)

    population.extend(offspring)
    lives = None if lives is None else np.pad(lives, (0, count), 'constant', constant_values=0)
    return population, lives
