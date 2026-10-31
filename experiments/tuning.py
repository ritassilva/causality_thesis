"""Run a resumable search, validation and held-out evaluation on final datasets.

Planning is the default. --execute runs fits in a new output directory; --resume
continues the same study. One PRISM study has the configured total trial budget.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from condition_data import load_sample, require_allowed_condition
from condition_data import result_table


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def suggest(trial, space, fixed):
    params = dict(fixed)
    for name, definition in space.items():
        if "when" in definition and params.get(definition["when"][0]) != definition["when"][1]:
            continue
        kind = definition["type"]
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, definition["choices"])
        elif kind == "int":
            params[name] = trial.suggest_int(name, int(definition["low"]), int(definition["high"]))
        else:
            params[name] = trial.suggest_float(name, definition["low"], definition["high"], log=kind == "float_log")
    return params


def specification(model, baseline, prism):
    if model != "PRISM":
        return dict(space=baseline["spaces"][model], defaults=baseline["defaults"][model],
                    fixed=baseline["fixed"].get(model, {}), trials=baseline["trials"][model], enqueue=[])
    return dict(space=prism["hpo"]["space"], defaults=None, fixed={}, trials=prism["hpo"]["n_trials"], enqueue=prism["hpo"]["enqueue"])


def fit_parameters(model, params, prism):
    if model != "PRISM":
        return {"params": params}
    from prism import load_defaults

    defaults = load_defaults()
    training = {key: value for key, value in prism["training"].items() if key in defaults["training"]}
    return dict(params=params, training=training,
                stein={k: v for k, v in prism["stein"].items() if k in defaults["stein"]},
                glasso={k: v for k, v in prism["glasso"].items() if k in defaults["glasso"]},
                bootstrap={k: v for k, v in prism["bootstrap"].items() if k in defaults["bootstrap"]},
                trace_every=prism["hpo"].get("fit_trace_every", 250))


def evaluate(model, params, stage, settings, prism, directory, *, ev_winner=None):
    dataset_study = "evaluation" if stage == "test" else "model_selection"
    thresholds = settings["threshold_grid"]
    rows = []
    for family in settings["families"]:
        require_allowed_condition(family)
        for seed in settings[f"{stage}_dataset_seeds"]:
            for model_seed in settings[f"{stage}_model_seeds"]:
                folder = directory / family / f"ds{seed}" / f"ms{model_seed}"
                fit_config = fit_parameters(model, params, prism)
                if stage == "test" and len(thresholds) == 1:
                    fit_config["threshold"] = thresholds[0]
                if model == "GOLEM" and params.get("equal_variances") is False:
                    if ev_winner is None:
                        raise ValueError("GOLEM-NV needs the selected EV configuration")
                    init_folder = settings["output"] / "GOLEM/ev_initialization" / family / f"ds{seed}"
                    init_path = init_folder / "initialization.npz"
                    if not init_path.exists():
                        ev_config = init_folder / "params.json"
                        write_json(ev_config, {"params": dict(ev_winner["params"], equal_variances=True), "threshold": 0.0})
                        command = [sys.executable, str(ROOT / "experiments/run_model.py"), "--study", dataset_study,
                                   "--condition", family, "--dataset-seed", str(seed), "--model", "GOLEM",
                                   "--model-seed", "0", "--arm", "tuned", "--params", str(ev_config),
                                   "--output", str(init_folder / "fit")]
                        completed = subprocess.run(command, capture_output=True, text=True, timeout=settings["timeout_seconds"])
                        if completed.returncode:
                            raise RuntimeError("GOLEM-EV initialization failed: " + completed.stderr[-2000:])
                        matrix = np.load(init_folder / "fit/prediction.npy", allow_pickle=False)
                        np.savez_compressed(init_path, B=matrix)
                    fit_config["params"] = dict(params, b_init_path=str(init_path))
                parameter_file = folder.parent / f"ms{model_seed}_params.json"
                write_json(parameter_file, fit_config)
                sweep_path = folder / "threshold_scores.npz"
                if not sweep_path.exists() and not (folder / "outcome.json").exists():
                    if folder.exists() and not (folder / "outcome.json").exists():
                        number = 1
                        interrupted = folder.with_name(folder.name + f"_interrupted_{number}")
                        while interrupted.exists():
                            number += 1
                            interrupted = folder.with_name(folder.name + f"_interrupted_{number}")
                        folder.rename(interrupted)
                    command = [sys.executable, str(ROOT / "experiments/run_model.py"), "--study", dataset_study,
                               "--condition", family, "--dataset-seed", str(seed), "--model", model,
                               "--model-seed", str(model_seed), "--arm", "tuned", "--params", str(parameter_file),
                               "--priors", "recompute", "--output", str(folder), "--threshold-grid", *map(str, thresholds)]
                    try:
                        completed = subprocess.run(command, capture_output=True, text=True, timeout=settings["timeout_seconds"])
                        folder.mkdir(parents=True, exist_ok=True)
                        (folder / "execution.log").write_text(completed.stdout + completed.stderr)
                        status = "finished" if completed.returncode == 0 and sweep_path.exists() else "error"
                    except subprocess.TimeoutExpired:
                        folder.mkdir(parents=True, exist_ok=True)
                        status = "timeout"
                    write_json(folder / "outcome.json", dict(status=status))
                if sweep_path.exists():
                    with np.load(sweep_path, allow_pickle=False) as cached:
                        scores = pd.DataFrame({key: cached[key] for key in cached.files})
                    run_info = json.loads((folder / "run.json").read_text())
                    for row in scores.to_dict(orient="records"):
                        row.update(model=model, dataset=family, dataset_seed=seed, model_seed=model_seed, status="finished", stage=stage,
                                   n_nodes=run_info["n_nodes"], n_samples=run_info["n_samples"],
                                   runtime_seconds=run_info["runtime_seconds"], arm="tuned")
                        rows.append(row)
                else:
                    status = json.loads((folder / "outcome.json").read_text())["status"]
                    # An unsuccessful fit scores zero at every candidate threshold.
                    for threshold in thresholds:
                        rows.append(dict(model=model, dataset=family, dataset_seed=seed, model_seed=model_seed,
                                         status=status, stage=stage, threshold=threshold, fbeta=0.0, is_acyclic=False))
    frame = pd.DataFrame(rows)
    frame["fbeta_failures_as_zero"] = frame.fbeta.fillna(0.0)
    if stage == "test":
        frame.loc[~frame.status.eq("finished"), "fbeta"] = np.nan
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "selection_scores.npz",
                        **{column: frame[column].to_numpy(dtype=str) if frame[column].dtype == object else frame[column].to_numpy() for column in frame})
    return frame


def choose_threshold(frame, require_complete=False, require_acyclic=False):
    if require_complete and not frame.status.eq("finished").all():
        return None
    candidates = []
    # Graph-output methods have a single, non-numeric threshold.
    graph_rows = frame.loc[frame.status.eq("finished") & frame.threshold.isna()]
    if len(graph_rows):
        frame = pd.concat([graph_rows, frame.loc[~frame.status.eq("finished")].drop_duplicates(["dataset", "dataset_seed", "model_seed"]).assign(threshold=np.nan)], ignore_index=True)
    for threshold, group in frame.groupby("threshold", dropna=False, sort=True):
        if require_acyclic and not group.is_acyclic.fillna(False).astype(bool).all():
            continue
        means = group.groupby(["dataset", "dataset_seed"]).fbeta.mean().groupby("dataset").mean()
        value = float(means.mean())
        candidates.append(dict(threshold=None if pd.isna(threshold) else float(threshold), value=value,
                               per_family={key: float(val) for key, val in means.items()}))
    return max(candidates, key=lambda value: value["value"]) if candidates else None


def evaluated_seed_offset(database, name):
    if not database.exists():
        return 0
    import optuna
    return len(optuna.load_study(study_name=name, storage="sqlite:///" + str(database)).trials)


def run_study(model, variant, settings, prism, spec, *, ev_winner=None):
    import optuna

    directory = settings["output"] / model / variant
    directory.mkdir(parents=True, exist_ok=True)
    fixed = dict(spec["fixed"])
    if model == "GOLEM":
        fixed["equal_variances"] = variant == "ev"
    if model == "AVICI":
        # Pretrained weights have no fitting hyperparameters; select only a threshold.
        params = dict(fixed)
        evaluate(model, params, "search", settings, prism, directory / "search")
        validation = evaluate(model, params, "validation", settings, prism, directory / "validation")
        result = choose_threshold(validation, require_complete=True)
        if result is None:
            raise RuntimeError("AVICI validation did not complete")
        winner = dict(params=params, threshold=result["threshold"], validation_value=result["value"], source_trial=None)
        write_json(directory / "winner.json", winner)
        return winner
    storage = "sqlite:///" + str(directory / "study.sqlite3")
    study = optuna.create_study(study_name=f"{model}_{variant}", storage=storage, direction="maximize", load_if_exists=True,
                               sampler=optuna.samplers.TPESampler(seed=settings["sampler_seed"] + evaluated_seed_offset(directory / "study.sqlite3", f"{model}_{variant}"), n_startup_trials=settings["n_startup_trials"], multivariate=model == "PRISM" and prism["hpo"].get("multivariate", True)))
    if not study.trials:
        enqueue = spec["enqueue"] or [spec["defaults"] or {}]
        for parameters in enqueue[:spec["trials"]]:
            study.enqueue_trial(parameters)

    def objective(trial):
        params = suggest(trial, spec["space"], fixed)
        trial.set_user_attr("fit_params", params)
        frame = evaluate(model, params, "search", settings, prism, directory / "search" / f"trial_{trial.number:04d}", ev_winner=ev_winner)
        result = choose_threshold(frame)
        trial.set_user_attr("selection", result)
        return result["value"] if result else 0.0

    evaluated = sum(trial.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED) for trial in study.trials)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.RUNNING:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            evaluated += 1
    remaining = max(0, spec["trials"] - evaluated)
    if remaining:
        study.optimize(objective, n_trials=remaining, callbacks=[lambda current, completed: setattr(current, "sampler", optuna.samplers.TPESampler(seed=settings["sampler_seed"] + len(current.trials), n_startup_trials=settings["n_startup_trials"], multivariate=model == "PRISM" and prism["hpo"].get("multivariate", True)))])
    candidates = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], key=lambda t: t.value, reverse=True)[:(prism["hpo"]["n_validation_candidates"] if model == "PRISM" else settings["n_validation_candidates"])]
    selected = []
    for trial in candidates:
        params = trial.user_attrs["fit_params"]
        frame = evaluate(model, params, "validation", settings, prism, directory / "validation" / f"trial_{trial.number:04d}", ev_winner=ev_winner)
        result = choose_threshold(frame, require_complete=True, require_acyclic=model == "PRISM")
        if result:
            selected.append(dict(params=params, threshold=result["threshold"], validation_value=result["value"], source_trial=trial.number))
    if not selected:
        raise RuntimeError(f"No admissible validation candidate for {model}/{variant}")
    winner = max(selected, key=lambda row: row["validation_value"])
    write_json(directory / "winner.json", winner)
    return winner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="PRISM, a baseline identifier, or all")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline_study.json")
    parser.add_argument("--prism-config", type=Path, default=ROOT / "configs/prism_study.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trials", type=int, help="Override total trials per study")
    parser.add_argument("--family", action="append", help="Restrict to named conditions")
    parser.add_argument("--variant", choices=["ev", "nv", "both"], default="both")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    baseline = json.loads(args.config.read_text())
    prism = json.loads(args.prism_config.read_text())
    if args.family:
        baseline["families"] = args.family
    for family in baseline["families"]:
        require_allowed_condition(family)
        for stage in ["search", "validation", "test"]:
            for seed in baseline[f"{stage}_dataset_seeds"]:
                load_sample("evaluation" if stage == "test" else "model_selection", family, seed)
    models = list(baseline["spaces"]) + ["PRISM"] if args.model == "all" else [args.model]
    if any(model not in baseline["spaces"] and model != "PRISM" for model in models):
        parser.error("Unknown model identifier")
    if args.trials is not None and args.trials < 1:
        parser.error("--trials must be positive")
    specs = {model: specification(model, baseline, prism) for model in models}
    if args.trials is not None:
        for spec in specs.values():
            spec["trials"] = args.trials
    if not args.execute:
        print(json.dumps(dict(models=models, specifications=specs, families=baseline["families"],
                              search_model_seeds=baseline["search_model_seeds"], validation_model_seeds=baseline["validation_model_seeds"],
                              test_model_seeds=baseline["test_model_seeds"]), indent=2))
        return
    configuration = dict(settings=baseline, prism=prism, specifications=specs, models=models, skip_test=args.skip_test, variant=args.variant)
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        old = json.loads((args.output / "configuration.json").read_text())
        if old["sha256"] != digest:
            raise ValueError("Resume requires the same configuration and trial budget")
    else:
        args.output.mkdir(parents=True)
        write_json(args.output / "configuration.json", dict(sha256=digest, **configuration))
    # A second process cannot resume or alter the same live study.
    import fcntl
    lock = (args.output / ".tuning.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another tuning process is using this output directory") from error
    settings = dict(baseline, output=args.output.resolve())
    winners = {}
    for model in models:
        if model == "GOLEM":
            # EV is selected first so NV always receives a genuine EV initialization.
            ev = run_study(model, "ev", settings, prism, specs[model])
            options = [ev]
            if args.variant in ("nv", "both"):
                nv = run_study(model, "nv", settings, prism, specs[model], ev_winner=ev)
                options = [nv] if args.variant == "nv" else [ev, nv]
            winner = max(options, key=lambda row: row["validation_value"])
        else:
            ev = None
            winner = run_study(model, "single", settings, prism, specs[model])
        winners[model] = winner
        if not args.skip_test:
            frozen_settings = dict(settings, threshold_grid=[winner["threshold"]] if winner["threshold"] is not None else [0.0])
            test = evaluate(model, winner["params"], "test", frozen_settings, prism, args.output / model / "test", ev_winner=ev)
            result_table(test.assign(arm="tuned")).to_csv(args.output / model / "test_results.csv", index=False)
        write_json(args.output / "winners.json", winners)


if __name__ == "__main__":
    main()
