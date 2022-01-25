import argparse
import itertools
import os

import numpy as np
import tqdm

import ray

from ReversiSimulator import evaluate_competition
from GeneticAlgoritm import initialize, sort_eliminate, crossover
from checkpoints import load_checkpoint, save_checkpoint
from agents import *


def generation_iter(start, end):
    start = 1 if start is None else start
    return itertools.count(start) if end is None else range(start, end + 1)


def result_iter(ray_references):
    while ray_references:
        done, ray_references = ray.wait(ray_references)
        yield ray.get(done[0])


def resize_population(population, lives, target_size):
    population_size = len(population)
    if population_size > target_size:
        population = population[:target_size]
        lives = None if lives is None else lives[:target_size]
    elif population_size < target_size:
        population_args = np.array([agent.to_list() for agent in population])
        population_bounds = np.max(population_args, axis=0), np.min(population_args, axis=0)
        population, lives = crossover(population, lives, target_size - population_size,
                                      bounds=population_bounds, mutate_rate=0.5,
                                      mutate_rfunc='gaussian', mutate_args=(0, np.std(population_args)))
    return population, lives


def evolve(agent_class_=DefaultAgent.AI, psize_=64, processors_=4,
           checkpoints_name_='checkpoints/default', start_gen_=None, end_gen_=None):
    ubound = np.array([-50, -50, -50, -50, -50, -50, -50, -50, -50, 0, 0, -50, -50, -50, -50, -50, -50]),
    lbound = np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 120, 50, 50, 50, 50, 50, 50])
    mrate = np.array([0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.45, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])

    load_return = () if start_gen_ is None else load_checkpoint(agent_class_, checkpoints_name_, start_gen_)
    if load_return:
        population, lives = resize_population(*load_return, target_size=psize_)
    else:
        population, lives = initialize(agent_class_, psize_, rfunc='uniform', rarg=(ubound, lbound))

    ray.init(ignore_reinit_error=True, num_cpus=processors_)
    remote_evaluation = ray.remote(evaluate_competition)
    print('Ray initialized')

    scores = np.zeros(psize_, dtype=np.intc)
    for generation in generation_iter(start_gen_, end_gen_):
        workers = [remote_evaluation.remote((i, population)) for i in range(psize_)]
        save_checkpoint(checkpoints_name_, generation, population, np.pad(scores, (0, psize_ - scores.size)), lives)
        scores = np.zeros(psize_, dtype=np.intc)
        for evaluate_result in tqdm.tqdm(result_iter(workers), total=psize_, desc=f'Iteration {generation}'):
            scores += evaluate_result
        population, scores, lives = sort_eliminate(population, scores, lives)
        bias = (np.argsort(scores) * 0.7 + np.argsort(lives) * 0.3) * (scores > 0) + psize_ * 0.5
        population, lives = crossover(population, lives, psize_ - len(population),
                                      bounds=(ubound, lbound), mutate_rate=mrate, select_bias=bias)


def make_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default=64, type=int)
    parser.add_argument('-s', '--start', default=None, type=int)
    parser.add_argument('-t', '--end', default=1000, type=int)
    parser.add_argument('--cpus', default=min(8, os.cpu_count()), type=int)
    return parser.parse_args()


if __name__ == '__main__':
    opt = make_opts()
    try:
        evolve(psize_=opt.size, processors_=opt.cpus, start_gen_=opt.start, end_gen_=opt.end)
    except KeyboardInterrupt:
        print('*End* Wait for ray processes shutting down...')
        pass
