import gc
import logging
import time

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, SGConv
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


SPARSE_ARCHS = {"sage", "gin", "sgc"}

RANKING_ARCHS = [{"name": n, "num_layers": L} for L in (4, 3, 2, 1) for n in ("gcn", "sage", "gat", "gin", "sgc")]

UTILITY_ARCHS = [{"name": n, "num_layers": 2} for n in ("gcn", "sage", "gat",  "sgc")]


class _GNN(torch.nn.Module):
    def __init__(self, layers, name):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.name = name
        self.sparse_ok = True

    def _run(self, x, ei):
        for i, conv in enumerate(self.layers):
            x = conv(x, ei)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def forward(self, x, edge_index, adj=None):
        use_sparse = (adj is not None and self.sparse_ok and self.name in SPARSE_ARCHS)
        if not use_sparse:
            return self._run(x, edge_index)
        try:
            return self._run(x, adj)
        except (NotImplementedError, RuntimeError, TypeError) as e:
            if "memory" in str(e).lower():
                raise
            self.sparse_ok = False
            logger.warning(f"{self.name}: sparse propagation unavailable ({e}) falling back to edge_index for this model.")
            return self._run(x, edge_index)


def build_gnn(name, in_dim, hidden, out_dim, num_layers=2):
    conv = {
        "gcn": lambda i, o: GCNConv(i, o),
        "sage": lambda i, o: SAGEConv(i, o),
        "gat": lambda i, o: GATConv(i, o),
        "sgc": lambda i, o: SGConv(i, o),
        "gin": lambda i, o: GINConv(Sequential(Linear(i, o), ReLU(), Linear(o, o))), }[name]
    dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
    return _GNN([conv(dims[k], dims[k + 1]) for k in range(num_layers)], name)


def attach_sparse_adj(data, enabled=True):
    """
    Build a CSR adjacency once per graph and stash it on the Data object.
    """
    if not enabled:
        data._adj = None
        return data
    try:
        ei = data.edge_index
        n = data.num_nodes
        vals = torch.ones(ei.size(1), device=ei.device)
        data._adj = torch.sparse_coo_tensor(ei, vals, (n, n)).coalesce().to_sparse_csr()
    except Exception as e:
        logger.warning(f"Sparse adjacency unavailable ({e}); using edge_index.")
        data._adj = None
    return data


def _adj_of(data):
    return getattr(data, "_adj", None)


@torch.no_grad()
def _evaluate(model, data, mask, metric):
    model.eval()
    if int(mask.sum()) == 0:
        return float("nan")
    logits = model(data.x, data.edge_index, _adj_of(data))[mask]
    y = data.y[mask]
    if metric == "acc":
        return (logits.argmax(1) == y).float().mean().item()
    prob1 = logits.softmax(1)[:, 1].detach().cpu().numpy()
    yt = y.cpu().numpy()
    if yt.min() == yt.max():
        return float("nan")
    return roc_auc_score(yt, prob1) if metric == "auroc" else average_precision_score(yt, prob1)


def fit_once(arch, data, masks, *, num_classes, metric="acc", seed=0, epochs=200, lr=0.01, wd=5e-4, hidden=64, patience=50, eval_every=10, task="multiclass"):
    """Train one model on using 'masks'. Returns the best-val model."""
    torch.manual_seed(seed)
    if data.x.is_cuda:
        torch.cuda.manual_seed_all(seed)
    device = data.x.device
    adj = _adj_of(data)

    weight = None
    if task != "multiclass":
        cnt = torch.bincount(data.y[masks["train"]], minlength=num_classes).float()
        weight = (cnt.sum() / cnt.clamp(min=1)).to(device)

    model = build_gnn(arch["name"], data.x.size(1), arch.get("hidden", hidden), num_classes, arch.get("num_layers", 2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val, best_state, bad = -float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(data.x, data.edge_index, adj)
        loss = F.cross_entropy(out[masks["train"]], data.y[masks["train"]], weight=weight)
        loss.backward()
        optimizer.step()
        del out, loss

        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            val = _evaluate(model, data, masks["val"], metric)
            val = -float("inf") if val != val else val
            if val > best_val:
                best_val, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += eval_every
                if bad >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    del optimizer, best_state
    return model, best_val



def budget_matched_masks(y_syn, y_real, real_masks, seed=0, name=""):
    """
    Give the synthetic graph the SAME number of labelled train/val nodes per class as the real split. 
    Used for SynGen
    """
    if y_syn.numel() == y_real.numel():
        return {k: v.clone() for k, v in real_masks.items()}, "real_masks"

    g = torch.Generator().manual_seed(seed)
    n = y_syn.numel()
    m = {k: torch.zeros(n, dtype=torch.bool) for k in ("train", "val", "test")}
    short = []
    for c in y_real.unique():
        n_tr = int((y_real[real_masks["train"]] == c).sum())
        n_va = int((y_real[real_masks["val"]] == c).sum())
        idx = (y_syn == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            short.append((int(c), 0, n_tr + n_va))
            continue
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        if idx.numel() < n_tr + n_va:
            short.append((int(c), int(idx.numel()), n_tr + n_va))
            n_tr = min(n_tr, idx.numel())
            n_va = max(0, min(n_va, idx.numel() - n_tr))
        m["train"][idx[:n_tr]] = True
        m["val"][idx[n_tr:n_tr + n_va]] = True
        m["test"][idx[n_tr + n_va:]] = True
    if short:
        logger.warning(f"{name}: fewer synthetic nodes than the real label budget for (class, available, needed) = {short}.")
        # This means some labels are not generated in the generative model. But nothing to do.
    return m, "budget_matched"


def _fit_and_eval(arch, train_data, train_masks, eval_data, eval_masks, *, num_classes, metric, seed, need_eval, **hp):
    """
    Fit one model and evaluate it. Returns {"test_self", "test_eval", "val"}:
      test_self: on the training graph's own test mask (TSTS / TRTR)
      test_eval: on eval_data's test mask (TSTR); None when not needed
    """
    model, val = fit_once(arch, train_data, train_masks, num_classes=num_classes, metric=metric, seed=seed, **hp)
    res = {
        "test_self": _evaluate(model, train_data, train_masks["test"], metric),
        "test_eval": (_evaluate(model, eval_data, eval_masks["test"], metric)
                      if (need_eval and eval_data is not None) else None),
        "val": float(val) if val == val and val != -float("inf") else float("nan"),
    }
    del model
    gc.collect()
    if train_data.x.is_cuda:
        torch.cuda.empty_cache()
    return res


def _key(arch):
    return f"{arch['name']}_L{arch.get('num_layers', 2)}_h{arch.get('hidden', 64)}"



def _mean_std(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, var ** 0.5


def _spearman(a, b):
    r = spearmanr(a, b)
    return float(getattr(r, "statistic", getattr(r, "correlation", float("nan"))))



def feature_space_flag(real_x, gen_x, sample=2000):
    """
    native           -- same dim, same value regime.
    mismatched_scale -- same dim, but one side is binary and the other is not
                        (GenCAT dense continuous vs binary bag-of-words).
    dim_mismatch     -- x exists but the spaces do not line up; ranking only.
    missing          -- no x at all.
    """
    if gen_x is None:
        return "missing"
    if real_x is None or gen_x.size(1) != real_x.size(1):
        return "dim_mismatch"
    a, b = real_x[:sample].float(), gen_x[:sample].float()
    ref_bin = bool((((a == 0) | (a == 1)).all()).item())
    gen_bin = bool((((b == 0) | (b == 1)).all()).item())
    if ref_bin != gen_bin:
        return "mismatched_scale"
    return "native"


def _fmt_eta(done, total, t0):
    if done <= 0:
        return "?"
    rate = (time.time() - t0) / done
    return f"{rate * max(0, total - done) / 60:.1f}m"


def _mem(device):
    if str(device) == "cpu" or not torch.cuda.is_available():
        return ""
    return (f", gpu {torch.cuda.memory_allocated()/2**30:.2f}G alloc / {torch.cuda.max_memory_allocated()/2**30:.2f}G peak")


def _release(data, device):
    data._adj = None
    data.to("cpu")
    gc.collect()
    if str(device) != "cpu":
        torch.cuda.empty_cache()


def downstream_all(baselines, masks, *,
                   utility_archs=None, ranking_archs=None,
                   utility_seeds=5, ranking_seeds=3, use_sparse_adj=True,
                   split_seed=0, task="multiclass", device="cpu", **hp):
    """
    baselines : {name: PyG Data}, must contain "Original".
    masks : train/val/test boolean masks on the real graph.
    split_seed : seed for sampling the budget-matched synthetic masks.
    """
    utility_archs = utility_archs or UTILITY_ARCHS
    ranking_archs = ranking_archs or RANKING_ARCHS
    metric = "acc" if task == "multiclass" else "auroc"

    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    real = attach_sparse_adj(baselines["Original"].to(device), use_sparse_adj)
    num_classes = int(real.y.max().item()) + 1
    real_masks = {k: v.to(device) for k, v in masks.items()}
    real_y_cpu = real.y.cpu()
    real_masks_cpu = {k: v.cpu() for k, v in real_masks.items()}

    logger.info(f"Downstream: real split has {int(real_masks['train'].sum())} train / {int(real_masks['val'].sum())} val / {int(real_masks['test'].sum())} "
                f"test nodes; synthetic training sets are budget-matched to it. sparse_adj={use_sparse_adj and _adj_of(real) is not None}")

    util_keys = {_key(a) for a in utility_archs}
    rank_keys = {_key(a) for a in ranking_archs}
    all_archs = list({_key(a): a for a in (list(ranking_archs) + list(utility_archs))}.values())

    def _seeds_for(arch):
        return max(ranking_seeds if _key(arch) in rank_keys else 0, utility_seeds if _key(arch) in util_keys else 0)

    n_models = sum(1 for k in baselines if k != "Original")
    per_model = (len(ranking_archs) * ranking_seeds + sum(max(0, utility_seeds - (ranking_seeds if k in rank_keys else 0)) for k in util_keys))
    total_fits = sum(_seeds_for(a) for a in all_archs) + n_models * per_model
    done, t0 = 0, time.time()

    util_rows, rank_rows, trtr = {}, {}, {}
    try:
        # TRTR
        for a in all_archs:
            for s in range(_seeds_for(a)):
                r = _fit_and_eval(a, real, real_masks, None, None, num_classes=num_classes, metric=metric, seed=s, need_eval=False, task=task, **hp)
                trtr[(_key(a), s)] = r["test_self"]
                done += 1
        logger.info(f"[Original] TRTR done, {done}/{total_fits} fits, elapsed {(time.time()-t0)/60:.1f}m{_mem(device)}")

        for name, g in baselines.items():
            if name == "Original":
                continue
            if getattr(g, "y", None) is None:
                logger.warning(f"{name}: no labels; skipped in both downstream tables.")
                continue
            if int(g.y.max().item()) >= num_classes:
                logger.warning(f"{name}: label id {int(g.y.max())} >= {num_classes}; skipped.")
                continue

            fflag = feature_space_flag(real.x, getattr(g, "x", None))
            if fflag == "missing":
                logger.warning(f"{name}: no node features; both tables N/A.")
                continue
            if fflag == "dim_mismatch":
                logger.warning(f"{name}: feature dim {g.x.size(1)} != {real.x.size(1)}; ranking only.")

            tstr_valid = fflag in ("native", "mismatched_scale")
            emit_util = tstr_valid
            if not tstr_valid:
                logger.info(f"{name}: TSTR N/A (features={fflag}).")

            g = attach_sparse_adj(g.to(device), use_sparse_adj)
            syn_masks_cpu, mask_mode = budget_matched_masks(
                g.y.cpu(), real_y_cpu, real_masks_cpu,
                seed=split_seed, name=name)
            syn_masks = {k: v.to(device) for k, v in syn_masks_cpu.items()}

            # Ranking and utility share (arch, seed) fits; fit each pair once.
            fits = {}
            def _fit(a, s):
                key = (_key(a), s)
                if key not in fits:
                    fits[key] = _fit_and_eval(a, g, syn_masks, real, real_masks, num_classes=num_classes, metric=metric, seed=s, need_eval=(tstr_valid and _key(a) in util_keys), task=task, **hp)
                return fits[key]

            # ranking
            per_seed_rank = []
            for s in range(ranking_seeds):
                tsts, trtr_vec = [], []
                for a in ranking_archs:
                    r = _fit(a, s)
                    tsts.append(r["test_self"])
                    trtr_vec.append(trtr[(_key(a), s)])
                    done += 1
                per_seed_rank.append({
                    "spearman": _spearman(trtr_vec, tsts),
                    "pearson": float(pearsonr(trtr_vec, tsts)[0]),
                })

            row = {}
            for k in ("spearman", "pearson"):
                m, sd = _mean_std([d[k] for d in per_seed_rank])
                row[k], row[k + "_std"] = m, sd
            row["n_seeds"] = ranking_seeds
            row["feature_space"] = fflag
            row["mask_mode"] = mask_mode
            rank_rows[name] = row

            # Utility
            if emit_util:
                for a in utility_archs:
                    ak = _key(a)
                    tstr_v, tsts_v, trtr_v, ratio_v = [], [], [], []
                    for s in range(utility_seeds):
                        r = _fit(a, s)
                        if not (ak in rank_keys and s < ranking_seeds):
                            done += 1
                        base = trtr[(ak, s)]
                        tstr_v.append(r["test_eval"] if tstr_valid else float("nan"))
                        tsts_v.append(r["test_self"])
                        trtr_v.append(base)
                        ratio_v.append((r["test_eval"] / max(base, 1e-9)) if tstr_valid else float("nan"))
                    out = {}
                    for label, vals in (("TSTR", tstr_v), ("TSTS", tsts_v), ("TRTR", trtr_v), ("ratio", ratio_v)):
                        m, sd = _mean_std(vals)
                        out[label], out[label + "_std"] = m, sd
                    out["TSTS_minus_TRTR"] = out["TSTS"] - out["TRTR"]
                    out["n_seeds"] = utility_seeds
                    out["feature_space"] = fflag
                    out["mask_mode"] = mask_mode
                    util_rows[f"{name}|{ak}"] = out

            _release(g, device)
            logger.info(f"[{name}] done, {done}/{total_fits} fits, ETA {_fmt_eta(done, total_fits, t0)}{_mem(device)}")

        # TRTR reference
        for a in utility_archs:
            ak = _key(a)
            m, sd = _mean_std([trtr[(ak, s)] for s in range(utility_seeds)])
            util_rows[f"Original|{ak}"] = {
                "TSTR": m, "TSTR_std": sd, "TSTS": m, "TSTS_std": sd,
                "TRTR": m, "TRTR_std": sd, "ratio": 1.0, "ratio_std": 0.0,
                "TSTS_minus_TRTR": 0.0, "n_seeds": utility_seeds,
                "feature_space": "native", "mask_mode": "real_masks",
            }
    finally:
        logger.info(f"Downstream: {done} fits, total {(time.time()-t0)/60:.1f}m{_mem(device)}")
        _release(real, device)

    return {"utility": util_rows, "ranking": rank_rows}
