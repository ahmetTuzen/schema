import os
import logging
import torch

import pandas as pd

from utils import report_fidelity, load_baselines    
    
from downstream_models import downstream_all, RANKING_ARCHS, UTILITY_ARCHS

logger = logging.getLogger(__name__)


def _cfg_get(cfg, dotted, default=None):
    cur = cfg
    for key in dotted.split("."):
        try:
            cur = getattr(cur, key)
        except Exception:
            return default
        if cur is None:
            return default
    return cur


def _infer_task(cfg):
    """If the label in yaml file is not binary_imbalanced, we are measuring accuracy for label classification"""
    t = _cfg_get(cfg, "evaluation.downstream.task", None)
    if t:
        return t
    return "multiclass"


def _make_split(y, train=0.6, val=0.2, seed=0):
    """Split by label -> boolean train/val/test masks."""
    g = torch.Generator().manual_seed(seed)
    n = y.size(0)
    m = {k: torch.zeros(n, dtype=torch.bool) for k in ("train", "val", "test")}
    for c in y.unique():
        idx = (y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.size(0), generator=g)]
        n_tr, n_va = int(train * idx.numel()), int(val * idx.numel())
        m["train"][idx[:n_tr]] = True
        m["val"][idx[n_tr:n_tr + n_va]] = True
        m["test"][idx[n_tr + n_va:]] = True
    return m


def _split_sig(cfg):
    return {
        "train_frac": float(_cfg_get(cfg, "evaluation.downstream.train_frac", 0.6)),
        "val_frac": float(_cfg_get(cfg, "evaluation.downstream.val_frac", 0.2)),
        "split_seed": int(_cfg_get(cfg, "evaluation.downstream.split_seed", 0)),
    }


def _get_masks(data, cfg):
    """
    Returns (masks, source_tag).
    The cache filename carries the split parameters.
    """
    has = lambda k: hasattr(data, k) and getattr(data, k) is not None
    ignore = bool(_cfg_get(cfg, "evaluation.downstream.ignore_dataset_split", False))
    if not ignore and has("train_mask") and has("test_mask") and (has("val_mask") or has("valid_mask")):
        val = data.val_mask if has("val_mask") else data.valid_mask
        return {"train": data.train_mask, "val": val, "test": data.test_mask}, "dataset"

    sig = _split_sig(cfg)
    split_dir = _cfg_get(cfg, "evaluation.downstream.split_dir", "splits")
    os.makedirs(split_dir, exist_ok=True)
    tag = f"tr{sig['train_frac']}_va{sig['val_frac']}_s{sig['split_seed']}"
    path = os.path.join(split_dir, f"{cfg.dataset.name}_split_{tag}.pt")
    if os.path.exists(path):
        return torch.load(path), "cached"

    masks = _make_split(data.y, train=sig["train_frac"], val=sig["val_frac"], seed=sig["split_seed"])
    torch.save(masks, path)
    logger.info(f"Created and cached the split -> {path}")
    return masks, "created"



def _resolve_device(cfg):
    dev = str(_cfg_get(cfg, "evaluation.downstream.device", "cuda"))
    if dev.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; downstream falls back to CPU (slow).")
        return "cpu"
    return dev


def _path(args, cfg, kind):
    return os.path.join(args.csv_location, f"downstream_{kind}", cfg.dataset.name + ".csv")


def downstream_fidelity(args, cfg):
    if args.evaluation_mode == "report":
        for kind in ("utility", "ranking"):
            path = _path(args, cfg, kind)
            if os.path.exists(path):
                report_fidelity(pd.read_csv(path), path)
        return

    logger.info("Calculating downstream fidelity...")
    results = calculate_downstream_fidelity(args, cfg)
    if not results:
        return
    logger.info("Downstream evaluation completed. Saving .csv(s), then reporting...")
    for kind, res in results.items():
        df, path = save_downstream_fidelity(args, cfg, res, kind)
        report_fidelity(df, path)


def calculate_downstream_fidelity(args, cfg):

    baselines = load_baselines(cfg)
    if "Original" not in baselines:
        logger.warning("Original graph not found... skipping downstream. Empty .csv will be created.")
        return {}

    masks, src = _get_masks(baselines["Original"], cfg)
    logger.info(f"Downstream split: {src}")

    hyperparameters = dict(
        epochs=int(_cfg_get(cfg, "evaluation.downstream.epochs", 200)),
        lr=float(_cfg_get(cfg, "evaluation.downstream.lr", 0.01)),
        wd=float(_cfg_get(cfg, "evaluation.downstream.weight_decay", 5e-4)),
        hidden=int(_cfg_get(cfg, "evaluation.downstream.hidden", 64)),
        patience=int(_cfg_get(cfg, "evaluation.downstream.patience", 50)),
        eval_every=int(_cfg_get(cfg, "evaluation.downstream.eval_every", 10)),
    )

    out = downstream_all(
        baselines, masks,
        utility_archs=_cfg_get(cfg, "evaluation.downstream.utility_archs", None) or UTILITY_ARCHS,
        ranking_archs=_cfg_get(cfg, "evaluation.downstream.ranking_archs", None) or RANKING_ARCHS,
        utility_seeds=int(_cfg_get(cfg, "evaluation.downstream.utility_seeds", 5)),
        ranking_seeds=int(_cfg_get(cfg, "evaluation.downstream.ranking_seeds", 3)),
        split_seed=_split_sig(cfg)["split_seed"],
        task=_infer_task(cfg),
        device=_resolve_device(cfg),
        **hyperparameters,
    )

    keep = {}
    if _cfg_get(cfg, "evaluation.downstream.utility", True):
        keep["utility"] = out["utility"]
    if _cfg_get(cfg, "evaluation.downstream.ranking", True):
        keep["ranking"] = out["ranking"]
    return keep


def save_downstream_fidelity(args, cfg, results, kind):
    save_path = _path(args, cfg, kind)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "model"
    df = df.reset_index()

    if args.evaluation_mode == "append" and os.path.exists(save_path):
        df_old = pd.read_csv(save_path)
        df = pd.concat([df_old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["model"], keep="last")

    df.to_csv(save_path, index=False)
    return df, save_path