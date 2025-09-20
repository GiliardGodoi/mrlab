import pytest
import yaml
from argparse import ArgumentParser
from typing import List
from dataclasses import dataclass, field
from mrlab.params import BaseParams

@dataclass
class Params(BaseParams):
    lr : float = None
    batch_size : int = None
    optimizer : str = None
    weights : List[float] = field(default_factory=list)
    logging_steps : str = None
    outputdir : str = None

    def template_folder_name(self):
        return "{outputdir}/{hash_id}"

@pytest.fixture
def config_file():
    return """
lr: !!float 5e-5
optimizer: AdamW
batch_size: 32
logging_steps: steps
weights: [1.0, 2.0]
"""

def test_config_file_reading(config_file):
    config = yaml.safe_load(config_file)
    print(config)
    params = Params(**config)
    assert params.lr == config.get('lr', None)
    assert isinstance(params.lr, float)
    assert isinstance(params.batch_size, int)
    assert isinstance(params.optimizer, str)
    assert isinstance(params.logging_steps, str)
    assert isinstance(params.weights, list)
    assert (params.get_default_folder()).exists()

def test_with_argument_parser():
    parser = ArgumentParser(description='Test cli')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-p', '--optimizer', type=str, default='SGD')
    parser.add_argument('-l', '--logging-steps', type=str, required=True)
    parser.add_argument('-w', '--weights', nargs=2, type=float, default=[-2.0, -2.0])

    args = parser.parse_args(
        "--lr 3e-5 -b 32 -l 500 -w 5.0 7.0".split()
    )
    params = Params(**vars(args))
    assert isinstance(params.weights, list)
    assert params.weights == [5.0, 7.0]
