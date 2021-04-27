import random
import pickle
import json
import numpy as np
import torch


def load_pickle(path: str):
    with open(path, "rb") as pkl_file:
        output = pickle.load(pkl_file)
    return output


def save_pickle(path: str, f: object) -> None:
    with open(path, "wb") as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_json(path: str, f: object) -> None:
    with open(path, "w") as json_path:
        json.dump(
            f,
            json_path,
        )

def load_json(path: str, encoding='utf-8') -> dict:
    with open(path, "r") as json_file:
        output = json.load(json_file)
    return output

def set_seed(seed: int = 42, contain_cuda: bool = False):
    random.seed(seed)
    np.random.seed(seed)

    if contain_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed set as {seed}")