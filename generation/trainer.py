import logging
import os
from typing import Dict, Optional

import torch

from checkpoint import _node_ckpt_path, _intra_ckpt_path, _inter_ckpt_path
from dataset_handler import NodeDataLoader, IntraEdgeDataLoader, InterEdgeDataLoader
from generator import _load_generated_x_for_intra
from model_builder import _build_node_generator, _build_intra_edge_model, _build_inter_edge_model

logger = logging.getLogger(__name__)


def _move_batch(batch: dict, device: str) -> dict:
    """Move tensors to device"""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(device) for t in v]
        else:
            out[k] = v
    return out


def node_training(args, skip_training: bool = False):
    """
    Trains the node generator on normalized features (continuous) or raw (binary). 
    """

    nodeloader = NodeDataLoader(args)
    dataloader = nodeloader.load_data()

    x_mean = nodeloader.x_mean.to(args.device)
    x_std = nodeloader.x_std.to(args.device)

    node_generator = _build_node_generator(args, dataloader, nodeloader.global_max_K)
    node_generator = node_generator.to(args.device)
    if hasattr(node_generator, 'init_weights'):
        node_generator.init_weights()
    is_binary = args.feature_type == 'binary'
    if is_binary and hasattr(node_generator, 'init_output_bias'):
        # x_mean is the per-feature base rate when features are binary
        node_generator.init_output_bias(x_mean)

    ckpt_path = _node_ckpt_path(args)

    # if skip training, load the checkpoint if it exists, otherwise train from scratch
    if skip_training and os.path.exists(ckpt_path):
        logger.info(f"[node_training] loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=True, map_location=args.device)
        node_generator.load_state_dict(ckpt['model_state'])
        x_mean = ckpt['x_mean'].to(args.device)
        x_std = ckpt['x_std'].to(args.device)
        return node_generator, dataloader, x_mean, x_std

    logger.info("[node_training] training from scratch")
    optimizer = torch.optim.Adam(node_generator.parameters(), lr=args.learning_rate)

    num_epochs = getattr(args, 'node_epochs', args.num_epochs)
    for e in range(num_epochs):
        node_generator.train()
        running = 0.0
        for i, batch in enumerate(dataloader):
            batch = _move_batch(batch, args.device)
            x_pool = batch['x_pool']
            S = batch['S']
            x = batch['x']
            mask = batch['mask']

            # Continuous features are trained in normalized space and denormalized by the assembler. 
            # Binary features are trained against the raw 0/1 targets with the model emitting logits.
            if is_binary:
                x_norm = x
            else:
                x_norm = (x - x_mean) / x_std.clamp(min=1e-8)

            optimizer.zero_grad()

            pred_norm = node_generator(x_pool, S, mask=mask)
            loss = node_generator.loss(pred_norm, x_norm, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(node_generator.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            if i % 20 == 0:
                logger.info(f"[node_training] epoch {e+1} batch {i} loss={loss.item():.4f}")

        logger.info(f"[node_training] epoch {e+1} avg loss={running / max(1, len(dataloader)):.4f}")

    torch.save({'model_state': node_generator.state_dict(), 'x_mean': x_mean.cpu(), 'x_std': x_std.cpu(),}, ckpt_path)
    logger.info(f"[node_training] saved checkpoint to {ckpt_path}")

    return node_generator, dataloader, x_mean, x_std


def _build_x_override_batch(
        batch: dict,
        x_generated: Dict[str, torch.Tensor],
        pad_value: float = 0.0,
    ) -> Optional[torch.Tensor]:
    """
    Build a [B, max_N, feat_dim] tensor of generated features aligned to the padded batch shape, for intra training.

    Returns None if the batch has no graph_id, a graph_id is missing from x_generated, 
    or a generated graph has fewer nodes than the batch expects.

    But this should not happen if the node generator is working correctly, so we log a warning in those cases. If happens, contact the authors.
    """
    # Pull shape + keys from the batch
    graph_ids = batch.get('graph_id', None)
    if graph_ids is None:
        return None

    B = batch['x'].shape[0]
    max_N = batch['x'].shape[1]
    fdim = batch['x'].shape[2]
    device = batch['x'].device
    node_mask = batch['node_mask'] # [B, max_N]

    out = torch.full((B, max_N, fdim), pad_value, device=device)

    for b in range(B):
        gid = graph_ids[b]
        if gid not in x_generated:
            logger.warning(f"[intra training] no generated x for {gid}, this should not be a case. Falling back to batch['x']")
            return None
        x_gen = x_generated[gid].to(device) # [N_real, fdim]
        N_real = int(node_mask[b].sum().item())
        if x_gen.shape[0] < N_real:
            logger.warning(f"[intra training] x_override for {gid} has {x_gen.shape[0]} nodes, batch expects {N_real}; falling back to batch['x']")
            return None
        out[b, :N_real] = x_gen[:N_real]

    return out


def intra_edge_training(args, x_mean: torch.Tensor, x_std: torch.Tensor, skip_training: bool = False):
    edge_loader = IntraEdgeDataLoader(args)
    dataloader = edge_loader.load_data()

    x_dim = dataloader.dataset[0]['x'].shape[1]
    edge_model = _build_intra_edge_model(args, x_dim).to(args.device)
    if hasattr(edge_model, 'init_weights'):
        edge_model.init_weights()

    ckpt_path = _intra_ckpt_path(args)

    if skip_training and os.path.exists(ckpt_path):
        logger.info(f"[intra training] loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=True, map_location=args.device)
        edge_model.load_state_dict(ckpt['model_state'])
        return edge_model

    logger.info("[intra training] training from scratch")
    optimizer = torch.optim.Adam(edge_model.parameters(), lr=args.learning_rate)
    num_epochs = getattr(args, 'intra_epochs', args.num_epochs)

    # Optional: use node-generator's reconstructed features instead of raw x
    x_generated: Optional[Dict[str, torch.Tensor]] = None
    if getattr(args, 'use_generated_x', False):
        x_gen_path = f"{args.out_dir}/{args.dataset}_{args.node_generator}_reconstructed.pt"
        if os.path.exists(x_gen_path):
            x_generated = _load_generated_x_for_intra(edge_loader, x_gen_path, x_mean, x_std)
        else:
            logger.warning(f"[intra training] use_generated_x=True but {x_gen_path} does not exist. Falling back to batch['x'].")

    for e in range(num_epochs):
        edge_model.train()
        running = 0.0
        for i, batch in enumerate(dataloader):
            batch = _move_batch(batch, args.device)
            
            # Generated features if available, otherwise the standardized reference x
            # (standardized for continuous and binary features alike)
            x_in = None
            if x_generated is not None:
                x_in = _build_x_override_batch(batch, x_generated)
            if x_in is None:
                x_in = (batch['x'] - x_mean) / x_std.clamp(min=1e-8)
            batch['x'] = x_in

            optimizer.zero_grad()
            loss = edge_model.loss(batch, x_override=None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(edge_model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            if i % 20 == 0:
                logger.info(f"[intra training] epoch {e+1} batch {i} loss={loss.item():.4f}")

        logger.info(f"[intra training] epoch {e+1} avg loss={running / max(1, len(dataloader)):.4f}")

    torch.save({'model_state': edge_model.state_dict()}, ckpt_path)
    logger.info(f"[intra training] saved checkpoint to {ckpt_path}")

    return edge_model


def inter_edge_training(args, x_mean: torch.Tensor, x_std: torch.Tensor, skip_training: bool = False):
    inter_loader = InterEdgeDataLoader(args)
    dataloader = inter_loader.load_data()

    sample = inter_loader.dataset[0]
    feat_dim = sample['x'].shape[1]
    pool_dim = sample['x_pool'].shape[-1]
    K = inter_loader.global_max_K

    model = _build_inter_edge_model(args, feat_dim, pool_dim, K).to(args.device)

    if hasattr(model, 'init_weights'):
        model.init_weights()

    ckpt_path = _inter_ckpt_path(args)

    if skip_training and os.path.exists(ckpt_path):
        logger.info(f"[inter training] loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=True, map_location=args.device)
        model.load_state_dict(ckpt['model_state'])
        return model, inter_loader

    logger.info("[inter training] training from scratch")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    num_epochs = getattr(args, 'inter_epochs', args.num_epochs)

    xm = x_mean.detach().cpu()
    xs = x_std.detach().cpu().clamp(min=1e-8)

    for e in range(num_epochs):
        model.train()
        running = 0.0
        for i, batch in enumerate(dataloader):
            # normalize on CPU in-place BEFORE the device transfer for the GPU memory 
            batch['x'] = batch['x'].sub_(xm).div_(xs)
            batch = _move_batch(batch, args.device) # handles edge_index_list

            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            if i % 20 == 0:
                logger.info(f"[inter training] epoch {e+1} batch {i} loss={loss.item():.4f}")

        logger.info(f"[inter training] epoch {e+1} avg loss={running / max(1, len(dataloader)):.4f}")

    torch.save({'model_state': model.state_dict()}, ckpt_path)
    logger.info(f"[inter training] saved checkpoint to {ckpt_path}")

    return model, inter_loader
