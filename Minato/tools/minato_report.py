#!/usr/bin/env python3
"""Summarise a Minato profile directory.

Reads the calibration JSON(s) and the per-process event CSVs written by
``MigrationRecorder`` and prints:

  * the calibrated t_out and the latency distribution it came from
  * migration rate, and where in the pipeline migrations happen
  * CPU actually wasted per migration under resume-from-index
  * the counterfactual: what the original "re-run the whole pipeline" scheme
    would have wasted for exactly the same migrations

Usage:  python3 tools/minato_report.py <profile_dir> [--json]
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np


def _load_calibration(profile_dir):
    best = None
    for path in sorted(glob.glob(os.path.join(profile_dir, "calibration_*.json"))):
        try:
            with open(path) as fh:
                blob = json.load(fh)
        except (OSError, ValueError):
            continue
        if best is None or blob.get("latency", {}).get("n", 0) > \
                best.get("latency", {}).get("n", 0):
            best = blob
    return best


def _load_events(profile_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(profile_dir, "events_*.csv"))):
        try:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        row["elapsed_s"] = float(row["elapsed_s"])
                        row["preserved_s"] = float(row["preserved_s"])
                        row["wasted_s"] = float(row["wasted_s"])
                        row["queue_wait_s"] = float(row["queue_wait_s"])
                        row["stage"] = int(row["stage"])
                        row["n_stages"] = int(row["n_stages"])
                        row["stage_times"] = json.loads(row["stage_times"] or "[]")
                    except (KeyError, ValueError, TypeError):
                        continue
                    rows.append(row)
        except OSError:
            continue
    return rows


def _fmt(x):
    return "%.4fs" % x


def build_report(profile_dir):
    calib = _load_calibration(profile_dir)
    events = _load_events(profile_dir)

    fast = [r for r in events if r["event"] == "fast"]
    migrations = [r for r in events if r["event"] == "migrate"]
    resumes = [r for r in events if r["event"] == "resume"]
    failures = [r for r in events if r["event"] == "resume_fail"]

    stage_median = {}
    pipeline_median = None
    if calib:
        for name, stats in (calib.get("stage_latency") or {}).items():
            stage_median[name] = stats.get("p50", 0.0)
        pipeline_median = calib.get("latency", {}).get("p50")

    stages = (calib or {}).get("stages") or []
    full_pipeline_cost = sum(stage_median.get(s, 0.0) for s in stages) or (pipeline_median or 0.0)

    at_stage = {}
    for r in migrations:
        at_stage[r["stage"]] = at_stage.get(r["stage"], 0) + 1

    wasted_now = sum(r["wasted_s"] for r in migrations)
    # Counterfactual (original artifact): the front worker's whole effort is
    # binned AND the background worker replays every stage from scratch.
    wasted_before = sum(r["elapsed_s"] for r in migrations) + \
        len(migrations) * full_pipeline_cost

    resumed_stage_counts = [r["stage"] for r in resumes]
    resume_time = sum(r["elapsed_s"] for r in resumes)
    skipped_cost = 0.0
    for r in resumes:
        skipped_cost += sum(stage_median.get(s, 0.0) for s in stages[:r["stage"]])

    report = {
        "profile_dir": os.path.abspath(profile_dir),
        "calibration": None,
        "counts": {
            "fast": len(fast),
            "migrated": len(migrations),
            "resumed": len(resumes),
            "resume_failures": len(failures),
        },
        "migration_rate": (len(migrations) / float(len(fast) + len(migrations))
                           if (fast or migrations) else 0.0),
        "migrations_at_stage": {str(k): v for k, v in sorted(at_stage.items())},
        "migrations_at_stage_0": at_stage.get(0, 0),
        "wasted_cpu_resume_from_index_s": wasted_now,
        "wasted_cpu_full_restart_s": wasted_before,
        "background_resume_time_s": resume_time,
        "background_work_skipped_s": skipped_cost,
        "pipeline_median_cost_s": full_pipeline_cost,
        "stages": stages,
    }
    if calib:
        report["calibration"] = {
            "t_out": calib.get("t_out"),
            "percentile": calib.get("percentile"),
            "source": calib.get("source"),
            "stop_reason": calib.get("stop_reason"),
            "wall_seconds": calib.get("calibration_wall_seconds"),
            "latency": calib.get("latency"),
            "stage_median": stage_median,
        }
    if migrations:
        report["mean_wasted_per_migration_s"] = wasted_now / len(migrations)
        report["mean_wasted_per_migration_full_restart_s"] = \
            wasted_before / len(migrations)
        if wasted_now > 0:
            report["waste_reduction_x"] = wasted_before / wasted_now
    if resumed_stage_counts:
        report["mean_resume_stage"] = float(np.mean(resumed_stage_counts))
    return report


def print_report(rep):
    print("=" * 72)
    print("Minato profile report  --  %s" % rep["profile_dir"])
    print("=" * 72)

    calib = rep.get("calibration")
    if calib:
        lat = calib.get("latency") or {}
        print("\n[1] Calibrated timeout")
        print("    t_out                 : %s  (P%.0f of the warm-up distribution)"
              % (_fmt(calib["t_out"]), calib["percentile"]))
        print("    source                : %s (%s)"
              % (calib.get("source"), calib.get("stop_reason")))
        if calib.get("wall_seconds"):
            print("    profiling wall time   : %.1fs" % calib["wall_seconds"])
        if lat.get("n"):
            print("    samples profiled      : %d" % lat["n"])
            print("    latency p50/p75/p90   : %s / %s / %s"
                  % (_fmt(lat["p50"]), _fmt(lat["p75"]), _fmt(lat["p90"])))
            print("    latency min/max       : %s / %s"
                  % (_fmt(lat["min"]), _fmt(lat["max"])))
        if calib.get("stage_median"):
            print("    per-stage median cost :")
            for name in rep["stages"]:
                print("        %-28s %s" % (name, _fmt(calib["stage_median"].get(name, 0.0))))
    else:
        print("\n[1] Calibrated timeout: no calibration_*.json found")

    c = rep["counts"]
    print("\n[2] Migration behaviour")
    print("    samples on fast path  : %d" % c["fast"])
    print("    samples migrated      : %d  (%.1f%% of samples)"
          % (c["migrated"], 100.0 * rep["migration_rate"]))
    print("    samples resumed in bg : %d   failures: %d"
          % (c["resumed"], c["resume_failures"]))
    if rep.get("mean_resume_stage") is not None:
        print("    mean resume stage     : %.2f of %d"
              % (rep["mean_resume_stage"], len(rep["stages"])))
    if rep["migrations_at_stage"]:
        print("    migrations by stage   :")
        for stage, n in rep["migrations_at_stage"].items():
            label = rep["stages"][int(stage)] if int(stage) < len(rep["stages"]) else "-"
            print("        after %s stage(s) -> resumes at %-24s %d"
                  % (stage, label, n))
    if rep.get("migrations_at_stage_0"):
        print("    NOTE: %d migration(s) happened before any stage finished, so"
              % rep["migrations_at_stage_0"])
        print("          those fell back to a full restart. If this dominates,")
        print("          your slowest stage is the first one and resume-from-index")
        print("          cannot help; split that stage to get finer checkpoints.")

    print("\n[3] Wasted CPU accounting")
    print("    resume-from-index     : %s total, %s per migration"
          % (_fmt(rep["wasted_cpu_resume_from_index_s"]),
             _fmt(rep.get("mean_wasted_per_migration_s", 0.0))))
    print("    original full-restart : %s total, %s per migration"
          % (_fmt(rep["wasted_cpu_full_restart_s"]),
             _fmt(rep.get("mean_wasted_per_migration_full_restart_s", 0.0))))
    if rep.get("waste_reduction_x"):
        print("    reduction             : %.1fx less duplicated CPU"
              % rep["waste_reduction_x"])
    print("    bg time spent resuming: %s" % _fmt(rep["background_resume_time_s"]))
    print("    bg work skipped       : %s (stages inherited from the front worker)"
          % _fmt(rep["background_work_skipped_s"]))
    print("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile_dir")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()
    if not os.path.isdir(args.profile_dir):
        print("no such directory: %s" % args.profile_dir, file=sys.stderr)
        return 2
    rep = build_report(args.profile_dir)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True, default=float))
    else:
        print_report(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
