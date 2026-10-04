import argparse
import torch
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    dataset = args.dataset.lower()
    root_dir = "data"

    if args.dataset == "cora_ml":
        from torch_geometric.datasets import CitationFull
        data = CitationFull(root=root_dir, name="Cora_ML")[0]
        torch.save(data, f"output/{dataset}_original.pt")
    elif dataset == "amazon_photo":
        from torch_geometric.datasets import Amazon
        data = Amazon(root=root_dir, name="Photo")[0]
        torch.save(data, f"output/{dataset}_original.pt")
    elif dataset == "amazon_computers":
        from torch_geometric.datasets import Amazon
        data = Amazon(root=root_dir, name="Computers")[0]
        torch.save(data, f"output/{dataset}_original.pt")
    elif dataset == "citeseer":
        from torch_geometric.datasets import Planetoid
        data = Planetoid(root=root_dir, name="CiteSeer")[0]
        torch.save(data, f"output/{dataset}_original.pt")
    elif dataset == "flickr":
        from torch_geometric.datasets import Flickr
        data = Flickr(root=f"{root_dir}/Flickr")[0]
        torch.save(data, f"output/{dataset}_original.pt")

    logger.info(f"[original_extractor] {dataset} .pt file saved to -> output/{dataset}_original.pt")


if __name__ == "__main__":
    main()