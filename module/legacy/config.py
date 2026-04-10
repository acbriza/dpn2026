import yaml


def load_config(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config