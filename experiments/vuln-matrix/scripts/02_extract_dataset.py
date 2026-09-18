#!/usr/bin/env python3
"""Script 2 -- dataset extractor / mixer.

    python3 scripts/02_extract_dataset.py \
        --profile manifests/openimages_profile.csv \
        --regime moderate --slow-fraction 0.25 -n 20000 \
        --out manifests/openimages_moderate_f0.25_n20000.csv

Emits exactly N rows drawn from the fast pool (P0-P50) and the slow pool
(regime-dependent) in the requested proportion, deterministically ordered, each
carrying its ground-truth pool label. The two sentinel samples are injected
into every manifest.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmp import manifest as M


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--regime", required=True, choices=sorted(M.RATIO_REGIMES))
    ap.add_argument("--slow-fraction", dest="slow_fraction", type=float, required=True)
    ap.add_argument("-n", "--n-samples", dest="n_samples", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dataset", default="", help="label recorded in metadata")
    ap.add_argument("--no-replacement", dest="no_replacement", action="store_true")
    a = ap.parse_args()

    rows = M.read_profile_manifest(a.profile)
    pmeta = M.read_profile_meta(a.profile)
    fast_rows, slow_rows, cuts = M.build_pools(rows, a.regime)
    sentinels = M.choose_sentinels(fast_rows, slow_rows)

    mixed, meta = M.mix_dataset(fast_rows, slow_rows, a.n_samples,
                                a.slow_fraction, a.seed, sentinels,
                                allow_replacement=not a.no_replacement)
    meta.update({
        "dataset": a.dataset or pmeta.get("pipeline", ""),
        "regime": a.regime, "cuts": cuts,
        "profile_manifest": os.path.abspath(a.profile),
        "profile_meta": pmeta,
        "fingerprint": M.manifest_fingerprint(mixed),
        "sentinel_detail": {k: (v and {"sample_id": v["sample_id"],
                                       "latency_s": v["latency_s"]})
                            for k, v in sentinels.items()},
    })
    M.write_run_manifest(a.out, mixed, meta)

    print("wrote %s" % a.out)
    print("  %d rows: %d fast + %d slow (slow_fraction=%.3f)"
          % (len(mixed), meta["n_fast"], meta["n_slow"],
             meta["slow_fraction_actual"]))
    print("  pools: fast=%d slow=%d  observed cost ratio=%s"
          % (cuts["fast_pool_size"], cuts["slow_pool_size"],
             round(cuts["observed_cost_ratio"], 2)
             if cuts.get("observed_cost_ratio") else "n/a"))
    print("  sentinels: %s" % json.dumps(meta["sentinel_ids"]))
    print("  fingerprint: %s" % meta["fingerprint"])
    for w in meta["warnings"]:
        print("  WARNING: %s" % w)
    if a.slow_fraction > 0.25:
        print("  NOTE: slow_fraction %.2f exceeds 1 - P75/100 = 0.25, so the "
              "warm-up P75 cut will land inside the slow pool. Expect a "
              "non-zero false-negative count; that is the effect under test, "
              "not a bug." % a.slow_fraction)


if __name__ == "__main__":
    main()
