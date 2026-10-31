"""Optional execution on a frozen sample; never used to reproduce saved figures."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "analysis")]
from condition_data import load_sample, load_sample_file, require_allowed_condition, sample_path, STUDY_FOLDERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--study", choices=["benchmark", "benchmark_control", "evaluation", "model_selection", "scalability", "ablation", "sachs", "tcga"])
    parser.add_argument("--condition")
    parser.add_argument("--dataset-seed", type=int, help="Required for conditions with multiple dataset seeds")
    parser.add_argument("--model", default="PRISM")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--arm", choices=["tuned", "untuned"], default="tuned")
    parser.add_argument("--priors", choices=["auto", "stored", "recompute"], default="auto")
    parser.add_argument("--threshold-grid", nargs="+", type=float, help="Score saved weights at each threshold after one fit")
    parser.add_argument("--params", type=Path, help="Explicit parameter JSON for a baseline")
    parser.add_argument(
        "--threshold", type=float, help="Explicit threshold for continuous baseline output"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New run directory; existing files are not overwritten",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.sample is not None and (args.study is not None or args.condition is not None):
        parser.error("Choose --sample or --study/--condition")
    if args.sample is None and not (args.study and args.condition):
        parser.error("Provide --sample or --study and --condition")
    require_allowed_condition(args.condition or args.sample)
    if args.sample:
        resolved = args.sample.resolve()
        for study, folder in STUDY_FOLDERS.items():
            if resolved.parent == (ROOT / "datasets" / folder).resolve():
                args.study, args.condition = study, resolved.stem
                break
        selected = load_sample_file(args.sample, args.dataset_seed)
    else:
        selected = load_sample(args.study, args.condition, args.dataset_seed)
    with selected as sample:
        X = np.asarray(sample["X"], float)
        truth = np.asarray(sample["true_adj"]) if "true_adj" in sample else None
        raw = np.asarray(sample["X_raw"], float) if "X_raw" in sample else None
        labels = (
            list(map(str, sample["col_names"]))
            if "col_names" in sample
            else [f"X{i}" for i in range(X.shape[1])]
        )
    if args.study == "sachs":
        from sachs_ground_truth import canonical_label, build_true_adj
        raw = X.copy() if raw is None else raw
        labels = [canonical_label(label) for label in labels]
        truth = build_true_adj(labels)
        scale = raw.std(0, keepdims=True)
        scale[scale == 0] = 1.0
        X = (raw - raw.mean(0, keepdims=True)) / scale
    random.seed(args.model_seed)
    np.random.seed(args.model_seed)
    try:
        import torch

        torch.manual_seed(args.model_seed)
        torch.set_num_threads(1)
    except ImportError:
        pass
    started = time.monotonic()
    if args.model == "PRISM":
        if args.arm != "tuned":
            raise ValueError(
                "PRISM is the frozen tuned model; use params for explicit ablations"
            )
        if raw is None:
            raise ValueError("PRISM needs the saved raw observations, not standardized data")
        from prism import PRISM

        implementation = PRISM
        prior_path = ROOT / "results/prism/priors/aligned" / f"{args.condition}__seed_{args.dataset_seed}.npz"
        use_stored = args.priors == "stored" or (args.priors == "auto" and args.study in ("evaluation", "ablation") and prior_path.is_file())
        if use_stored:
            with np.load(prior_path, allow_pickle=False) as cached:
                frozen_priors = {"r_G": cached["r_G"], "r_H": cached["r_H_raw"],
                                 "binary": cached["d_binary"], "bootstrap": cached["d_bootstrap"]}
            class StoredPriorPRISM(PRISM):
                def compute_priors(self, observations):
                    return {key: value.copy() for key, value in frozen_priors.items()}
            implementation = StoredPriorPRISM
        overrides = json.loads(args.params.read_text()) if args.params else {}
        if args.threshold is not None:
            overrides["threshold"] = args.threshold
        model = implementation(seed=args.model_seed, **overrides).fit(raw)
        prediction = model.adjacency_
        continuous = False
        weights = model.weights_
        config = dict(model.config, threshold=model.threshold, prior_source="stored" if use_stored else "recomputed")
    elif args.model == "Random":
        from random_baseline import sample_random_dag

        prediction = sample_random_dag(X.shape[1], seed=args.model_seed)
        continuous = False
        weights = None
        config = {"edge_probability": 0.5}
    else:
        from baseline_runners import run

        if args.params:
            config = json.loads(args.params.read_text())
            params = config.get("params", config)
        elif args.arm == "tuned":
            config = json.loads((ROOT / "configs/frozen/frozen_winners.json").read_text())[
                "winners"
            ][args.model]
            params = config["params"]
        else:
            params = json.loads((ROOT / "configs/frozen/untuned_defaults.json").read_text())[
                args.model
            ]
            config = {"params": params, "threshold": None}
        if (
            args.model == "GOLEM"
            and params.get("equal_variances") is False
            and "b_init_path" not in params
            and args.study in {"benchmark", "benchmark_control", "evaluation", "model_selection", "ablation", "scalability"}
        ):
            family = args.condition or args.sample.stem.split("__seed_")[0]
            dataset_seed = args.dataset_seed if args.dataset_seed is not None else int(args.sample.stem.split("__seed_")[1])
            init = ROOT / f"results/tuned/golem_initialization/{family}/ds{dataset_seed}.npz"
            if not init.exists():
                raise FileNotFoundError(f"Frozen GOLEM EV initialization required: {init}")
            params = dict(params, b_init_path=str(init))
        prediction, continuous = run(args.model, X, labels, params)
        weights = None
    args.output.mkdir(parents=True)
    if isinstance(prediction, np.ndarray):
        np.save(args.output / "prediction.npy", prediction)
    else:
        np.save(args.output / "endpoint_matrix.npy", prediction.graph)
    if weights is not None:
        np.save(args.output / "weights.npy", weights)
    if truth is not None:
        from metrics import compute_metrics, fbeta_score, directed_auprc

        threshold = args.threshold if args.threshold is not None else config.get("threshold")
        if threshold is None and continuous and args.arm == "untuned":
            threshold = json.loads((ROOT / "configs/experiment_protocol.json").read_text())["untuned_thresholds"].get(args.model)
        if not continuous or threshold is not None:
            graph = prediction
            if continuous:
                graph = (np.abs(np.asarray(prediction)) > threshold).astype(int)
                np.fill_diagonal(graph, 0)
            scores = compute_metrics(truth, graph, labels)
            precision, recall = scores.get("precision"), scores.get("recall")
            if precision is not None and recall is not None:
                scores["fbeta"] = fbeta_score(precision, recall)
            if continuous or weights is not None:
                auprc = directed_auprc(truth, np.abs(weights if weights is not None else prediction))
                (args.output / "auprc.json").write_text(json.dumps({"auprc": auprc}) + "\n")
            (args.output / "metrics.json").write_text(json.dumps(scores, indent=2) + "\n")
    if truth is not None and args.threshold_grid:
        import pandas as pd
        import networkx as nx
        from metrics import compute_metrics, to_adjacency_matrix, fbeta_score, _make_label_to_internal
        sweep = []
        matrix = weights if weights is not None else prediction
        has_weights = weights is not None or continuous
        thresholds = args.threshold_grid if has_weights else [None]
        for threshold_value in thresholds:
            graph = (np.abs(np.asarray(matrix)) > threshold_value).astype(int) if has_weights else prediction
            if has_weights:
                np.fill_diagonal(graph, 0)
            scores = compute_metrics(truth, graph, labels)
            precision, recall = scores.get("precision"), scores.get("recall")
            if precision is None or recall is None:
                raise ValueError("Precision/recall could not be calculated for the fitted graph")
            scores["fbeta"] = fbeta_score(precision, recall)
            adjacency, _ = to_adjacency_matrix(graph, node_order=labels, label_to_internal=_make_label_to_internal(labels))
            scores.update(threshold=threshold_value, is_acyclic=nx.is_directed_acyclic_graph(nx.from_numpy_array(adjacency, create_using=nx.DiGraph)))
            sweep.append(scores)
        sweep_frame = pd.DataFrame(sweep)
        np.savez_compressed(args.output / "threshold_scores.npz",
                            **{column: sweep_frame[column].to_numpy(dtype=float) for column in sweep_frame})
    (args.output / "run.json").write_text(
        json.dumps(
            dict(
                model=args.model,
                model_seed=args.model_seed,
                arm=args.arm,
                sample=str(args.sample) if args.sample else str(sample_path(args.study, args.condition)),
                study=args.study,
                condition=args.condition,
                dataset_seed=args.dataset_seed,
                n_nodes=int(X.shape[1]),
                n_samples=int(X.shape[0]),
                config=config,
                continuous=continuous,
                runtime_seconds=time.monotonic() - started,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
