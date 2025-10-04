import pandas as pd
import numpy as np
import pytest
import yaml
from typing import List
from dataclasses import dataclass, field
from mrlab.search import GridSearch
from mrlab.params import BaseParams

@dataclass
class Params(BaseParams):
    lr:float = None
    batch_size:int = None
    optimizer:str = None
    logging_steps:str = None
    outputdir:str = None

    def template_folder_name(self):
        return "{outputdir}/{hash_id}"

@dataclass
class ParamsWithList(BaseParams):
    lr: float = None
    batch_size:int = None
    weights: List[float] = field(default_factory=list)

    def template_folder_name(self):
        return "{hash_id}"

@pytest.fixture
def search_space():
    return {
        "lr": [1e-5, 5e-5, 1e-4],
        "batch_size": [4, 8],
        "optimizer": ["AdamW", "SGD"],
        "logging_steps" : 'steps'
    }

def simulate_training(params):
    cols = list('ABC')
    df = pd.DataFrame(
        np.random.rand(5, 3),
        columns=cols
    )
    folder = params.dir_metrics()
    df.to_csv(folder / 'metrics.csv')
    return True

def test_to_yaml(tmp_path):
    params = Params(
        lr=1e-5,
        batch_size=8,
        optimizer='Nesterov',
        outputdir=str(tmp_path)
    )
    f = params.to_yaml()
    assert f.exists()
    args = Params.from_yaml(f)
    assert args.hash_id == params.hash_id
    assert args._hash_id == args.hash_id
    assert args._timestamp == params._timestamp
    assert args.lr == params.lr
    assert args.batch_size == params.batch_size
    assert args.optimizer == params.optimizer
    assert args.base_folder == params.base_folder
    assert args.dir_checkpoints() == params.dir_checkpoints()
    assert args.dir_metrics() == params.dir_metrics()
    assert args.dir_logs() == params.dir_logs()
    assert args.dir_images() == params.dir_images()
    assert args.dir_predictions() == params.dir_predictions()

def test_to_json(tmp_path):
    params = Params(
        lr=1e-5,
        batch_size=8,
        optimizer='Nesterov',
        outputdir=str(tmp_path)
    )
    f = params.to_json()
    assert f.exists()
    args = Params.from_json(f)
    assert args.hash_id == params.hash_id
    assert args._hash_id == args.hash_id
    assert args.lr == params.lr
    assert args.batch_size == params.batch_size
    assert args.optimizer == params.optimizer
    assert args.base_folder == params.base_folder
    assert args.dir_checkpoints() == params.dir_checkpoints()
    assert args.dir_metrics() == params.dir_metrics()
    assert args.dir_logs() == params.dir_logs()
    assert args.dir_images() == params.dir_images()
    assert args.dir_predictions() == params.dir_predictions()

def test_grid_search_with_parameters(search_space, tmp_path):
    initial = Params(outputdir=str(tmp_path))
    grid = GridSearch(search_space, initial=initial)
    for params in grid:
        f = params.to_yaml()
        assert f.exists()
        args = Params.from_yaml(f)
        assert args and (args.hash_id == params.hash_id)

def test_grid_search_look_up(search_space, tmp_path):
    initial = Params(outputdir=str(tmp_path))
    grid = GridSearch(search_space, initial=initial)
    for params in grid:
        assert simulate_training(params)


def test_params_with_lists(tmp_path):
    content = '''
    lr : 0.1
    batch_size: 68
    weights :
        - 2.0
        - 5.0
    '''
    config = yaml.safe_load(content)
    params = ParamsWithList(**config)

    assert isinstance(params.weights, list)
    assert params.weights == [2.0, 5.0]
