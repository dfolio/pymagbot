import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def save_to_json(data: dict, filepath: str):
    """Save dictionary to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def load_from_json(filepath: str) -> dict:
    """Load dictionary from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_to_yaml(data: dict, filepath: str):
    """Save dictionary to a YAML file."""
    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def load_from_yaml(filepath: str) -> dict:
    """Load dictionary from a YAML file."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def save_to_csv(data: dict, filepath: str):
    """Save dictionary to a CSV file using pandas."""
    df = pd.DataFrame([data])  # Convert dict to DataFrame
    df.to_csv(filepath, index=False)


def load_from_csv(filepath: str) -> dict:
    """Load dictionary from a CSV file using pandas."""
    df = pd.read_csv(filepath)
    return df.to_dict(orient="records")[0]  # Convert first row to dict


def save_to_npz(data: dict, filepath: str):
    """Save dictionary to a NumPy .npz file."""
    np.savez(filepath, **data)


def load_from_npz(filepath: str) -> dict:
    """Load dictionary from a NumPy .npz file."""
    npzfile = np.load(filepath, allow_pickle=True)
    return {key: npzfile[key].tolist() for key in npzfile.files}
