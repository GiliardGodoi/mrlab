import hashlib
import json
import yaml
from dataclasses import (
    asdict,
    dataclass,
    field
)
from pathlib import Path

@dataclass
class BaseArguments:

    _hash_id : str = field(init=False, default=None, repr=False)

    def update(self, **kwargs):
        """Cria uma nova instância com valores atualizados"""
        base_dict = asdict(self)
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