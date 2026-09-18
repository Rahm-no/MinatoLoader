#!/usr/bin/env python3
"""Script 0 -- write a synthetic JPEG corpus with heavy-tailed decode cost.

Exists so the Mac smoke test needs no dataset download. The images are noise,
but they are real JPEGs of realistically varying dimensions, so the decode
stage shows the same cost heterogeneity that drives the OpenImages result.
Never use this for numbers destined for the paper.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmp.pipelines import make_synthetic_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heavy-frac", dest="heavy_frac", type=float, default=0.15)
    a = ap.parse_args()
    paths = make_synthetic_corpus(a.out, a.n, a.seed, heavy_frac=a.heavy_frac)
    total = sum(os.path.getsize(p) for p in paths)
    print("wrote %d jpegs to %s (%.1f MB)" % (len(paths), a.out, total / 1e6))


if __name__ == "__main__":
    main()
