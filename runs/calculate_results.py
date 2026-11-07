"""Calculate and save numerical notebook results; figures stay in notebooks."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'analysis'), str(ROOT / 'runs')]
import pandas as pd
import calculations
from synthetic_data import configure_samples


def main():
    functions = {
        'priors': calculations.calculate_priors,
        'tuning': calculations.calculate_tuning,
        'statistics': calculations.calculate_statistics,
        'benchmark': calculations.calculate_benchmark,
        'sortability': calculations.calculate_sortability,
        'tcga_biological': calculations.calculate_tcga_biological,
        'tcga_agreement': calculations.calculate_tcga_agreement,
        'sachs': calculations.calculate_sachs,
        'scalability': calculations.calculate_scalability,
        'parameters': calculations.calculate_parameters,
        'ablation': calculations.calculate_ablation,
        'families': calculations.calculate_families,
        'convergence': calculations.calculate_convergence,
        'plot_data': calculations.calculate_plot_data,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, help="JSON sample-file manifest")
    parser.add_argument('--section', choices=list(functions), action='append')
    args = parser.parse_args()
    if args.samples:
        configure_samples(args.samples)
    output = ROOT / 'results/analysis'
    output.mkdir(exist_ok=True)
    sections = args.section or list(functions)
    needs_samples = {"priors", "tuning", "sortability"}
    if not args.samples and needs_samples.intersection(sections):
        parser.error("Recalculating priors, tuning or sortability requires --samples.")
    for section in sections:
        print('Calculating', section, flush=True)
        tables = functions[section]()
        pd.to_pickle(tables, output / f'{section}.pkl')
        print('Saved', section, flush=True)


if __name__ == '__main__':
    main()
