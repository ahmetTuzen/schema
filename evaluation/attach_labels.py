import argparse
import glob
import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _attach_labels(target_path, y, force) -> bool:
    """
    Attaches labels to Schema outputs (v8_x.pt)
    """
    try:
        target = torch.load(target_path, weights_only=False)
    except Exception as e:
        logger.warning(f"[attach_labels] Skipping -> {target_path}: load failed ({e})")
        return False

    if not hasattr(target, 'original_node_indices'):
        logger.warning(f"[attach_labels] Skipping -> {target_path}: no original_node_indices")
        return False

    idx = target.original_node_indices.long()
    existing = getattr(target, 'y', None)
    if existing is not None and existing.shape[0] == idx.shape[0] and not force:
        logger.info(f"[attach_labels] Not changed -> {os.path.basename(target_path)}: has labels and no --force.")
        return False

    if idx.numel() and idx.max().item() >= y.shape[0]:
        logger.warning(f"[attach_labels] Skipping -> {target_path}: max index {idx.max().item()} >= {y.shape[0]} reference nodes")
        return False
    
    target.y = y[idx]
    tmp = target_path + ".tmp"
    torch.save(target, tmp)
    os.replace(tmp, target_path)
    logger.info(f"[attach_labels] Completed-> {os.path.basename(target_path)}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--original', required=True, help='Reference PyG Data .pt file')
    ap.add_argument('--targets-dir', required=True, help='Directory scanned recursively for generated samples')
    ap.add_argument('--force', action='store_true', help='Overwrite existing labels')
    args = ap.parse_args()

    if not os.path.exists(args.original):
        logger.error(f"[attach_labels] Original not found: {args.original}")
        return

    logger.info(f"[attach_labels] Loading original: {args.original}")

    original = torch.load(args.original, weights_only=False)
    y = getattr(original, 'y', None)
    if y is None:
        logger.error(f"[attach_labels] {args.original} has no labels (y)")
        return

    targets = sorted({p for pat in ('*_v8_*.pt') for p in glob.glob(os.path.join(args.targets_dir, '**', pat), recursive=True)})
    if not targets:
        logger.error(f"[attach_labels] No generated samples found under {args.targets_dir}")
        return
            
    logger.info(f"[attach_labels] Processing {len(targets)} sample(s)...")
    n_updated = sum(_attach_labels(t, y, args.force) for t in targets)
    logger.info(f"[attach_labels] Finished. {n_updated}/{len(targets)} updated.")


if __name__ == "__main__":
    main()