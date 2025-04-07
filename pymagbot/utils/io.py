import os
import json

import numpy as np
import pandas as pd
import yaml

from pymagbot.utils import DEFAULT_FMT


def save_to(data, filepath: str, fmt: str = DEFAULT_FMT) -> None:
    """Save data to file"""
    if fmt == "json":
        save_to_json(data, filepath)
    elif fmt == "yaml":
        save_to_yaml(data, filepath)
    elif fmt == "csv":
        save_to_csv(data, filepath)
    raise NotImplementedError(f"Save to format {fmt} not implemented")


def load_from(filepath: str, fmt: str = None) -> dict:
    """Load data from file"""
    if fmt is None:
        base_name = os.path.basename(filepath)
        name, fmt = os.path.splitext(base_name)
        
    if fmt == "json":
        return load_from_json(filepath)
    elif fmt == "yaml":
        return load_from_yaml(filepath)
    elif fmt == "csv":
        return load_from_csv(filepath)
    raise NotImplementedError(f"Load from format {fmt} not implemented")


def save_to_json(data: (dict, str), filepath: str):
    """Save dictionary to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def load_from_json(filepath: str) -> dict:
    """Load dictionary from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_to_yaml(data: (dict, str), filepath: str):
    """Save dictionary to a YAML file."""
    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def load_from_yaml(filepath: str) -> dict:
    """Load dictionary from a YAML file."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def save_to_csv(data: (dict, str), filepath: str):
    """Save dictionary to a CSV file using pandas."""
    df = pd.DataFrame([data])  # Convert dict to DataFrame
    df.to_csv(filepath, index=False)


def load_from_csv(filepath: str) -> dict:
    """Load dictionary from a CSV file using pandas."""
    df = pd.read_csv(filepath)
    return df.to_dict(orient="records")[0]  # Convert first row to dict


def save_to_npz(data: (dict, str), filepath: str):
    """Save dictionary to a NumPy .npz file."""
    np.savez(filepath, **data)


def load_from_npz(filepath: str) -> dict:
    """Load dictionary from a NumPy .npz file."""
    npzfile = np.load(filepath, allow_pickle=True)
    return {key: npzfile[key].tolist() for key in npzfile.files}
