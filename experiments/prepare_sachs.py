"""Materialize the complete observational Sachs sample and 17-arc reference."""
from pathlib import Path
import argparse
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"analysis"))
from condition_data import load_sample
from sachs_ground_truth import canonical_label,build_true_adj
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    sample=load_sample("sachs", "sachs_cd3cd28")
    raw=sample["X"];labels=[canonical_label(c) for c in sample["col_names"]]
    if raw.shape!=(853,11):raise ValueError('Unexpected Sachs sample shape')
    scale=raw.std(axis=0,ddof=0);scale[scale==0]=1
    X=(raw-raw.mean(axis=0))/scale
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,X=X,X_raw=raw,true_adj=build_true_adj(labels),col_names=np.asarray(labels))

if __name__=='__main__':main()
