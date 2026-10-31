"""Generate a new dataset using the pinned causalgen package.

The historical final benchmark uses the included frozen samples. Generating
new data with the current package does not replace those historical samples.
"""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    options = json.loads(args.config.read_text())
    if re.search(r"mod[_-]?subtype", json.dumps(options), flags=re.I):
        raise ValueError("ModSubtype is excluded from this experiment collection")
    from causalgen import GeneratorConfig, generate
    data = generate(GeneratorConfig(**options))
    data.save(args.output, prefix="dataset", include_plot=False)


if __name__ == "__main__":
    main()
