import torch
import os
from torch_geometric.seed import seed_everything


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed all RNGs via PyG's seed_everything (covers random, numpy, torch, torch.cuda). 
    Optional deterministic mode enables determinism at a significant performance cost.
    """

    seed_everything(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)