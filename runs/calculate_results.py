"""Calculate and save numerical notebook results; figures stay in notebooks."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'analysis'), str(ROOT / 'runs')]
import pandas as pd
import calculations


def main():
    functions = {
        'priors': calculations.calculate_priors,
        'tuning': calculations.calculate_tuning,
        'statistics': calculations.calculate_statistics,
        'benchmark': calculations.calculate_benchmark,
        'control': calculations.calculate_control,
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
    parser.add_argument('--section', choices=list(functions), action='append')
    args = parser.parse_args()
    output = ROOT / 'results/analysis'
    output.mkdir(exist_ok=True)
    for section in args.section or functions:
        print('Calculating', section, flush=True)
        tables = functions[section]()
        pd.to_pickle(tables, output / f'{section}.pkl')
        print('Saved', section, flush=True)


if __name__ == '__main__':
    main()
