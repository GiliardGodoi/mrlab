

import pytest
from dataclasses import dataclass
from tempfile import TemporaryDirectory
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

@pytest.fixture
def params():
    folder = TemporaryDirectory()
    return Params(
        lr=1e-5,
        batch_size=8,
        optimizer='Nesterov',
        outputdir=folder.name
    )

@pytest.fixture
def args(params):
    f = params.to_yaml()
    assert f.exists()  # garante que o arquivo foi criado
    return Params.from_yaml(f)


def test_hash_id(args, params):
    assert args.hash_id == params.hash_id

def test_timestamp(args, params):
    assert args._timestamp == params._timestamp

def test_lr(args, params):
    assert args.lr == params.lr

def test_batch_size(args, params):
    assert args.batch_size == params.batch_size

def test_optimizer(args, params):
    assert args.optimizer == params.optimizer

def test_base_folder(args, params):
    assert args.base_folder == params.base_folder

def test_dir_checkpoints(args, params):
    assert args.dir_checkpoints() == params.dir_checkpoints()

def test_dir_metrics(args, params):
    assert args.dir_metrics() == params.dir_metrics()

def test_dir_logs(args, params):
    assert args.dir_logs() == params.dir_logs()

def test_dir_images(args, params):
    assert args.dir_images() == params.dir_images()

def test_dir_predictions(args, params):
    assert args.dir_predictions() == params.dir_predictions()
