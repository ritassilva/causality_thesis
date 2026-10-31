"""Plan or explicitly execute benchmark fits on stored final samples.

Planning is the default. Existing publication results are never overwritten.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from condition_data import load_results, load_sample, sample_path
from condition_data import result_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=lambda value: {"main": "benchmark", "final": "tuned"}.get(value, value),
                        choices=["benchmark", "tuned", "control"], default="tuned")
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
    study = {"benchmark": "benchmark", "tuned": "evaluation", "control": "benchmark_control"}[args.study]
    frame = load_results(study, n_variables=args.nodes)
    rows = frame.to_dict(orient="records")
    tasks = []
    available_seeds = {}
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
        sample = sample_path(study, family)
        if family not in available_seeds:
            import numpy as np
            with np.load(sample, allow_pickle=False) as observations:
                available_seeds[family] = set(observations["dataset_seeds"].astype(int))
        if seed not in available_seeds[family]:
            raise ValueError(f"Dataset seed {seed} not found for {family}")
        tasks.append(
            dict(
                dataset=family,
                dataset_seed=seed,
                model=model,
                model_seed=ms,
                arm=arm,
                sample=str(sample.relative_to(ROOT)),
                study=study,
            )
        )
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
            str(ROOT / "experiments/run_model.py"),
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
    import pandas as pd
    result_table(pd.DataFrame(outcomes)).to_csv(args.output / "results.csv", index=False)
    if any(row["status"] != "finished" for row in outcomes):
        raise SystemExit("Some fits failed; inspect results.csv and per-fit execution.log")


if __name__ == "__main__":
    main()
