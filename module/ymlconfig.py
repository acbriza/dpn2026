import yaml

from types import SimpleNamespace

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d