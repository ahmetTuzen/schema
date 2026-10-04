import re

LABEL_POLICY = {
    "er_p": "none",
    "er_m": "none",
    "ws": "none",
    "ba": "none",
    "cl": "copy_degree",
    "config": "copy_degree",
    "dcsbm": "copy_edge",
    "gae_recon": "copy_edge",
    "vgae_prior": "none",
    "cell": "copy_edge",
    "netgan": "copy_edge",
    "sagess": "copy_edge",
    "gencat": "generated",
    "graphmaker": "generated",
    "syngen": "generated",
    "lgsg": "none",
    "schema": "copy_edge",
}

NODE_ALIGN = {
    "er_p": "none",
    "er_m": "none",
    "ws": "none",
    "ba": "none",
    "cl": "degree",
    "config": "degree",
    "dcsbm": "edge",
    "gae_recon": "edge",
    "vgae_prior": "none",
    "cell": "edge",
    "netgan": "edge",
    "sagess": "edge", 
    "gencat": "degree",
    "graphmaker": "none",
    "syngen": "none",
    "lgsg": "none", 
    "schema": "edge",
}


GENERATES_FEATURES = {"gencat", "graphmaker", "syngen", "schema"}

ALIGNED = ("edge", "degree")

_SAMPLE_SUFFIX = re.compile(r"[#_]\d+$")


def strip_sample_suffix(name):
    """'cell#3' -> 'cell'. Used for yaml files."""
    return _SAMPLE_SUFFIX.sub("", str(name))


def infer_model(name):
    """
    Extract node alignment for the model.
    """
    hay = strip_sample_suffix(name).lower()
    for k in sorted(NODE_ALIGN, key=len, reverse=True):
        if k in hay:
            return k
    return None


def node_align(model, default="unknown"):
    return NODE_ALIGN.get(model, default)


def label_policy(model, default="none"):
    return LABEL_POLICY.get(model, default)


def generates_features(model):
    return model in GENERATES_FEATURES


def is_aligned(align):
    return align in ALIGNED