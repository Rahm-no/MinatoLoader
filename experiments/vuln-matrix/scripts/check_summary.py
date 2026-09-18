#!/usr/bin/env python3
"""Fail loudly if the summary CSV has empty or impossible fields.

The point of the smoke test is to catch a metric that silently never got
populated, which is invisible until you try to plot the full matrix.
"""
import csv, sys

REQUIRED_ALL = ["run_id", "loader", "n_samples", "epoch_wall_s",
                "throughput_samples_s", "sample_p50_s", "sample_p95_s",
                "batch_mean_s", "batch_p95_s"]
REQUIRED_MINATO = ["t_out_s", "warmup_n", "warmup_p50_s", "migrations",
                   "migrations_at_stage_0", "detect_tax_total_s",
                   "tp_slow_migrated", "fn_slow_kept_fast", "fp_fast_migrated",
                   "tn_fast_kept_fast", "misclassification_rate", "slow_recall",
                   "wasted_cpu_resume_s", "batch_composition_skew"]


def main(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("FAIL: summary.csv is empty"); return 1

    problems = []
    for r in rows:
        need = list(REQUIRED_ALL)
        if r["loader"] == "minato":
            need += REQUIRED_MINATO
        for k in need:
            v = (r.get(k) or "").strip()
            if v == "" or v.lower() in ("nan", "none"):
                problems.append("%s: %s is empty" % (r["run_id"], k))
        try:
            if float(r["throughput_samples_s"]) <= 0:
                problems.append("%s: non-positive throughput" % r["run_id"])
        except (ValueError, KeyError):
            pass
        if int(float(r.get("n_errors") or 0)) > 0:
            problems.append("%s: %s sample errors during the run"
                            % (r["run_id"], r["n_errors"]))
        if r["loader"] == "minato":
            tot = sum(int(float(r[k] or 0)) for k in
                      ("tp_slow_migrated", "fn_slow_kept_fast",
                       "fp_fast_migrated", "tn_fast_kept_fast"))
            if tot == 0:
                problems.append("%s: confusion matrix is all zeros" % r["run_id"])

    print("checked %d run(s) in %s" % (len(rows), path))
    for r in rows:
        print("  %-28s %-8s thr=%-8.8s  p95_batch=%-8.8s  t_out=%-8.8s "
              "recall=%-6.6s" % (r["run_id"][-28:], r["loader"],
                                 r.get("throughput_samples_s", ""),
                                 r.get("batch_p95_s", ""), r.get("t_out_s", "-"),
                                 r.get("slow_recall", "-")))
    if problems:
        print("\nFAIL:")
        for p in problems:
            print("  " + p)
        return 1
    print("\nOK: every required field is populated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "summary.csv"))
