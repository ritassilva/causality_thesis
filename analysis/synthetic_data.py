"""Resolve sample files by study and dataset identifier."""
import json
from pathlib import Path

_SAMPLES = {}


def configure_samples(manifest):
    """Load sample records from a JSON manifest."""
    path = Path(manifest).resolve()
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError("The samples manifest must be a JSON list")
    samples = {}
    for row in records:
        key = (_study(row['study']), str(row['dataset']), int(row['dataset_seed']))
        if key in samples:
            raise ValueError(f"Duplicate sample identity: {key}")
        source = Path(row['sample']).expanduser()
        source = (path.parent / source).resolve() if not source.is_absolute() else source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        samples[key] = source
    _SAMPLES.clear()
    _SAMPLES.update(samples)


def _study(study):
    if study in {"search", "validation", "model_selection"}:
        return "model_selection"
    if study in {"tuned", "tuned_untuned", "evaluation", "ablation"}:
        return "evaluation"
    return study


def sample_path(dataset, dataset_seed, *, study="evaluation", project_root=None):
    """Return the sample path for a study and dataset identifier."""
    key = (_study(study), str(dataset), int(dataset_seed))
    if key not in _SAMPLES:
        raise FileNotFoundError(
            f"No input file supplied for {key}; provide it in --samples."
        )
    return _SAMPLES[key]
