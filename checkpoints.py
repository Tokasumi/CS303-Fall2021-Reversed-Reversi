import json
import typing
from pathlib import Path

import numpy as np


def save_checkpoint(name: str, gen, population: list, scores: typing.Iterable, lives: typing.Iterable = None):
    if lives is None:
        try:
            lives = [agent.life for agent in population]
        except AttributeError:
            lives = [0] * len(population)
    assert np.shape(population) == np.shape(scores) == np.shape(
        lives), f'{np.shape(population)} {np.shape(scores)} {np.shape(lives)}'
    checkpoint_path = Path(name)
    if not checkpoint_path.exists():
        checkpoint_path.mkdir()
    with checkpoint_path.joinpath(Path(f'gen_{gen}.json')).open('w') as json_file:
        write_population = []
        for i, agent in enumerate(population):
            write_population.append({
                'args': [float(x) for x in agent.to_list()],
                'score': int(scores[i]),
                'life': int(lives[i])
            })
        json.dump(obj={'population': write_population}, fp=json_file)


def load_checkpoint(agent_class, name: str, gen):
    checkpoint = Path(name).joinpath(Path(f'gen_{gen}.json'))
    if not checkpoint.exists():
        return False
    with checkpoint.open('r') as json_file:
        population_info = json.load(fp=json_file)
        population,  lives = [], []
        for agent_dict in population_info['population']:
            agent = agent_class()
            agent.from_list(agent_dict['args'])
            lives.append(agent_dict['life'])
            population.append(agent)
    return population, lives
