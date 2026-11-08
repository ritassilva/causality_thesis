"""Plan or execute benchmark fits."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from synthetic_data import configure_samples, sample_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=lambda value: {"main": "benchmark", "final": "tuned"}.get(value, value),
                        choices=["benchmark", "tuned"], default="tuned")
    parser.add_argument("--samples", type=Path, required=True, help="JSON sample-file manifest")
    parser.add_argument("--nodes", type=int, help="Restrict benchmark graph size; omit for all stored sizes")
    parser.add_argument("--model", help="Restrict to one model identifier")
    parser.add_argument("--family", help="Restrict to one dataset family")
    parser.add_argument("--arm", choices=["tuned", "untuned"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset-seed", type=int, action="append")
    parser.add_argument("--model-seed", type=int, action="append")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    configure_samples(args.samples)
    study = {"benchmark": "benchmark", "tuned": "evaluation"}[args.study]
    import pandas as pd
    if study == "evaluation":
        frame = pd.concat([pd.read_csv(path, low_memory=False, float_precision="round_trip") for path in sorted((ROOT / "results/tuned").glob("*.csv"))], ignore_index=True)
    else:
        frame = pd.concat(
            [pd.read_csv(path, low_memory=False, float_precision="round_trip")
             for path in sorted((ROOT / "results/benchmark").glob("*.csv"))],
            ignore_index=True,
        )
    frame["model"] = frame.model.replace({"DAGMA_Reg": "PRISM", "DAGMA_Prior": "PRISM"})
    frame["arm"] = frame["arm"].fillna("untuned") if "arm" in frame else "untuned"
    frame["n_nodes"] = frame.n_nodes.fillna(pd.to_numeric(frame.dataset.str.extract(r"^(\d+)nodes", expand=False)))
    if args.nodes is not None:
        frame = frame.loc[frame.n_nodes.eq(args.nodes)]
    # Stored CSV rows select configurations
    protocol = json.loads((ROOT / "configs/experiment_protocol.json").read_text())
    model_seeds = protocol["model_seeds"][study]
    if args.model_seed and not set(args.model_seed).issubset(model_seeds):
        parser.error(f"--model-seed must belong to {model_seeds}")
    identities = ["dataset", "model", "arm", "dataset_seed"]
    metadata = [c for c in ("condition_id",) if c in frame]
    frame = frame[identities + metadata].drop_duplicates(identities).merge(
        pd.DataFrame({"model_seed": model_seeds}), how="cross"
    )
    rows = frame.to_dict(orient="records")
    tasks = []
    for row in rows:
        family = row["dataset"]
        model = row["model"].replace("PRISM", "PRISM")
        arm = row.get("arm") or "untuned"
        if args.model and model != args.model:
            continue
        if args.family and family != args.family:
            continue
        if args.arm and arm != args.arm:
            continue
        seed = int(row["dataset_seed"])
        ms = int(row["model_seed"])
        if args.dataset_seed and seed not in args.dataset_seed:
            continue
        if args.model_seed and ms not in args.model_seed:
            continue
        sample = sample_path(family, seed, study=study)
        if not sample.is_file():
            raise FileNotFoundError(sample)
        tasks.append(
            dict(
                dataset=family,
                dataset_seed=seed,
                model=model,
                model_seed=ms,
                arm=arm,
                sample=str(sample),
                study=study,
            )
        )
        for column in metadata:
            if pd.notna(row[column]):
                tasks[-1][column] = row[column]
    if not tasks:
        raise ValueError("No tasks match the selection")
    if args.output.exists():
        plan_file = args.output / "plan.json"
        reusable = plan_file.is_file() and json.loads(plan_file.read_text()) == tasks
        if not reusable or (not args.resume and any(p.name != "plan.json" for p in args.output.iterdir())):
            raise FileExistsError(args.output)
    else:
        args.output.mkdir(parents=True)
    (args.output / "plan.json").write_text(json.dumps(tasks, indent=2) + "\n")
    print(f"{len(tasks)} scheduled fits; execute={args.execute}")
    if not args.execute:
        return
    outcomes = []
    for task in tasks:
        directory = (
            args.output
            / task["arm"]
            / task["dataset"]
            / f"ds{task['dataset_seed']}"
            / task["model"]
            / f"ms{task['model_seed']}"
        )
        if directory.exists() and args.resume:
            if (directory / "metrics.json").is_file() and (directory / "run.json").is_file():
                scores = json.loads((directory / "metrics.json").read_text())
                information = json.loads((directory / "run.json").read_text())
                outcomes.append(dict(task, status="finished", returncode=0, **scores,
                                     runtime_seconds=information["runtime_seconds"], n_nodes=information["n_nodes"], n_samples=information["n_samples"]))
                continue
            number = 1
            previous = directory.with_name(directory.name + f"_incomplete_{number}")
            while previous.exists():
                number += 1
                previous = directory.with_name(directory.name + f"_incomplete_{number}")
            directory.rename(previous)
        command = [
            sys.executable,
            str(ROOT / "runs/run_model.py"),
            "--sample",
            task["sample"],
            "--study",
            task["study"],
            "--condition",
            task["dataset"],
            "--dataset-seed",
            str(task["dataset_seed"]),
            "--model",
            task["model"],
            "--model-seed",
            str(task["model_seed"]),
            "--arm",
            task["arm"],
            "--output",
            str(directory),
        ]
        if task["model"] == "PRISM" and task["study"] == "evaluation":
            command += ["--priors", "stored"]
        if task["model"] == "GOLEM":
            init = ROOT / f"results/tuned/golem_initialization/{task['dataset']}/ds{task['dataset_seed']}.npz"
            if init.is_file():
                command += ["--golem-init", str(init)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
            row = dict(
                task,
                status="finished" if result.returncode == 0 else "error",
                returncode=result.returncode,
            )
            directory.mkdir(exist_ok=True, parents=True)
            (directory / "execution.log").write_text(result.stdout + result.stderr)
            metrics = directory / "metrics.json"
            if metrics.exists():
                row.update(json.loads(metrics.read_text()))
            elif result.returncode == 0:
                row["status"] = "error"
                row["error"] = "The fit did not produce scored metrics"
            run_info = directory / "run.json"
            if run_info.exists():
                information = json.loads(run_info.read_text())
                row.update({key: information[key] for key in ["runtime_seconds", "n_nodes", "n_samples"]})
        except subprocess.TimeoutExpired:
            row = dict(task, status="timeout")
        outcomes.append(row)
    measurements = pd.DataFrame(outcomes).assign(timed_out=lambda frame: frame.status.eq("timeout"))
    if study in {"benchmark", "evaluation"}:
        # Keep public result CSVs compact; detailed diagnostics remain in
        # per-fit metrics.json, run.json and execution.log.
        columns = ["dataset", "model", "arm", "dataset_seed"]
        columns += ["model_seed", "condition_id", "status", "timed_out", "n_nodes",
                    "n_samples", "f1", "fbeta", "precision", "recall", "SHD", "runtime_seconds"]
        measurements = measurements.reindex(columns=columns)
    measurements.to_csv(args.output / "results.csv", index=False)
    if any(row["status"] != "finished" for row in outcomes):
        raise SystemExit("Some fits failed; inspect results.csv and per-fit execution.log")


if __name__ == "__main__":
    main()
