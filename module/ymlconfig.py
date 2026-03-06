import yaml

from types import SimpleNamespace

def load_config(path="config.yml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

config_dict = load_config()
config = dict_to_namespace(config_dict)