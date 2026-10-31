"""Optional PRISM execution using frozen TCGA data and priors."""

import argparse
import json
import random
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"analysis"))
from condition_data import load_sample

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    from prism import load_defaults
    from prism.penalty import build_lambda
    from tcga_mask import build_model, verify

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with load_sample("tcga", args.dataset) as data:
        X = data["X"]
        labels = data["col_names"].astype(str)
        kinds = data["kinds"].astype(str)
    with np.load(ROOT / f"results/tcga/priors/{args.dataset}.npz", allow_pickle=False) as data:
        if not np.array_equal(labels, data["col_names"].astype(str)):
            raise ValueError("Prior label order differs")
        priors = {
            "r_G": data["r_G"],
            "r_H": data["r_H_raw"],
            "binary": data["d_binary"],
            "bootstrap": data["d_bootstrap"],
        }
    cfg = load_defaults()
    penalty = build_lambda(priors, cfg["params"])
    allowed = None
    if args.dataset.startswith("tcga_mut_expr70"):
        allowed = ~((kinds == "expression")[:, None] & (kinds == "mutation")[None, :])
        np.fill_diagonal(allowed, False)
    random.seed(args.model_seed)
    np.random.seed(args.model_seed)
    torch.manual_seed(args.model_seed)
    torch.set_num_threads(1)
    train = dict(cfg["training"])
    hidden = train.pop("hidden_dim")
    model = build_model(X.shape[1], hidden, allowed)
    weights = model.fit(X, penalty, **train)
    constraint = verify(weights, allowed)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "weights.npz", W=weights, col_names=labels, kinds=kinds)
    (args.output / "run.json").write_text(
        json.dumps(
            dict(
                dataset=args.dataset,
                model_seed=args.model_seed,
                config=cfg,
                constraint_check=constraint,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
