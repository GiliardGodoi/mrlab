import hashlib
import json
import re
import yaml
from collections.abc import Iterable
from dataclasses import (
    asdict,
    dataclass,
    field,
    fields
)
from datetime import datetime
from pathlib import Path
from typing import List

def options(values):
    if not isinstance(values, Iterable):
        raise TypeError(f'Values options must e a iterable: got {type(values)}')
    return field(default_factory=lambda: values)

@dataclass
class BaseParams:

    _hash_id : str = field(default=None, repr=False)
    _timestamp: str = field(default_factory=lambda: datetime.now().isoformat(), repr=False)

    def _check_attrs_names(self, **kwargs):

        valid_fields = {f.name for f in fields(self)}
        invalid_args = set(kwargs) - valid_fields
        if invalid_args:
            raise TypeError(
                f"Invalid argument(s) for {self.__class__.__name__}:"
                f"{invalid_args}"
                f"\n\tIt is not allowed to instantiate an object with an argument not defined in parameter class."
            )

    def update(self, **kwargs):
        """Cria uma nova instância com valores atualizados"""
        self._check_attrs_names(**kwargs)
        base_dict = asdict(self)
        del base_dict['_hash_id']
        base_dict.update(kwargs)
        return self.__class__(**base_dict)

    # -------------------------
    # Leitura de arquivo
    # -------------------------
    @classmethod
    def from_dict(cls, config : dict) :
        return cls(**config)

    @classmethod
    def from_json(cls, filepath:'str'):
        with open(filepath, 'r') as f:
            config = json.load(f)
        return cls.from_dict(config)

    @classmethod
    def from_yaml(cls, filepath:'str'):
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    # -------------------------
    # Conversão
    # -------------------------
    def to_dict(self):
        return asdict(self)

    def to_yaml(self):
        filepath = self.get_default_folder() / 'arguments.yaml'
        with open(filepath, 'w') as f:
            txt = yaml.dump(self.to_dict())
            f.write(txt)
        return filepath

    def to_json(self):
        filepath = self.get_default_folder() / 'arguments.json'
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, sort_keys=True)
        return filepath

    # -------------------------
    # Hash dos argumentos
    # -------------------------
    def make_hash_id(self):
        args_dict = asdict(self)
        del args_dict['_hash_id']
        args_txt = json.dumps(args_dict, sort_keys=True)
        digest = hashlib.md5(args_txt.encode()).hexdigest()
        return digest

    @property
    def hash_id(self):
        if self._hash_id is None:
            self._hash_id = self.make_hash_id()
        return self._hash_id

    # -------------------------
    # Diretório para os argumentos
    # -------------------------
    def template_folder_name(self):
        raise NotImplementedError()
        # return "{hassh_id}"

    def get_base_folder_name_from_arguments(self):
        template = self.template_folder_name()
        args = asdict(self)
        args['hash_id'] = self.hash_id
        return Path(template.format(**args))

    def get_default_folder(self, key=None, ensure_exists=True):
        if key is None:
            folder = self.get_base_folder_name_from_arguments()
        else:
            folder = self.get_base_folder_name_from_arguments() / key
        if ensure_exists:
            folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def base_folder(self):
        return self.get_base_folder_name_from_arguments()

    # -------------------------
    # Pastas específicas
    # -------------------------
    def dir_checkpoints(self, ensure_exists=True):
        return self.get_default_folder('checkpoints', ensure_exists)

    def dir_logs(self, ensure_exists=True):
        return self.get_default_folder('logs', ensure_exists)

    def dir_metrics(self, ensure_exists=True):
        return self.get_default_folder('metrics', ensure_exists)

    def dir_predictions(self, ensure_exists=True):
        return self.get_default_folder('predictions', ensure_exists)

    def dir_images(self, ensure_exists=True):
        return self.get_default_folder('images', ensure_exists)

@dataclass
class MyParameters(BaseParams):

    lr: float = 0.0
    batch_size: int = 0
    optimizer_name: str = 'Nostorov'
    penalization_weights: List[float] = options([0.1, 0.3, 0.5])
    logging_steps: str ='steps'

    def template_folder_name(self):
        return "outputs/{hash_id}"