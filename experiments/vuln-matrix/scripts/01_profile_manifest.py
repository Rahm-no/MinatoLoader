#!/usr/bin/env python3
"""Script 1 -- profiling manifest builder.

One plain pass over a candidate pool with a fixed seed and a small fixed worker
count, recording per sample: id, path, encoded size, total preprocessing
latency, and per-transform timings.

    python3 scripts/01_profile_manifest.py \
        --root /dev/shm/openimages --pipeline openimages \
        --out manifests/openimages_profile.csv --limit 60000

Reproducibility: the seed is written into <out>.meta.json together with the
pipeline, its kwargs, the stage names and the host. Two manifests with the same
meta block are directly comparable; ones with different meta blocks are not.
"""
import argparse, glob, json, os, platform, socket, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmp import manifest as M
from hmp import pipelines
from hmp.minato import run_stages

AUDIO_EXT = (".flac", ".wav", ".mp3", ".opus")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")


def discover(root, exts, limit, seed):
    import random
    files = []
    for dirpath, _dirs, names in os.walk(root):
        for n in sorted(names):
            if n.lower().endswith(exts):
                files.append(os.path.join(dirpath, n))
    files.sort()
    if not files:
        raise SystemExit("no files matching %s under %s" % (exts, root))
    if limit and limit < len(files):
        random.Random(seed).shuffle(files)
        files = sorted(files[:limit])
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="directory of candidate samples")
    ap.add_argument("--pipeline", required=True, choices=sorted(pipelines.REGISTRY))
    ap.add_argument("--pipeline-kwargs", dest="pipeline_kwargs", default="{}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = every file found")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--repeats", type=int, default=1,
                    help="time each sample N times and keep the median")
    ap.add_argument("--progress-every", dest="progress_every", type=int, default=500)
    a = ap.parse_args()

    exts = AUDIO_EXT if a.pipeline == "librispeech" else IMAGE_EXT
    files = discover(a.root, exts, a.limit, a.seed)
    kwargs = json.loads(a.pipeline_kwargs)
    stages = pipelines.build_stages(a.pipeline, kwargs, seed=a.seed)
    print("profiling %d samples through %d stages: %s"
          % (len(files), len(stages), [n for n, _ in stages]))

    rows, failures = [], 0
    t_start = time.time()
    for i, path in enumerate(files):
        sid = os.path.relpath(path, a.root)
        try:
            size = os.path.getsize(path)
            samples, stage_times = [], []
            for r in range(a.repeats):
                timings = []
                t0 = time.perf_counter()
                run_stages(stages, {"row": {"sample_id": sid, "path": path}},
                           0, None, None, timings)
                samples.append(time.perf_counter() - t0)
                stage_times = timings
            samples.sort()
            rows.append({
                "sample_id": sid, "path": path, "bytes": size,
                "latency_s": "%.6f" % samples[len(samples) // 2],
                "stage_times_json": json.dumps(
                    [[n, round(d, 6)] for n, d in stage_times]),
                "extra_json": json.dumps({"repeats": a.repeats}),
            })
        except Exception as exc:
            failures += 1
            if failures <= 10:
                print("  skip %s: %r" % (sid, exc))
        if a.progress_every and (i + 1) % a.progress_every == 0:
            rate = (i + 1) / (time.time() - t_start)
            print("  %d/%d  (%.1f samples/s)" % (i + 1, len(files), rate))

    meta = {
        "pipeline": a.pipeline, "pipeline_kwargs": kwargs, "seed": a.seed,
        "stages": [n for n, _ in stages], "root": os.path.abspath(a.root),
        "n_candidates": len(files), "n_profiled": len(rows),
        "failures": failures, "repeats": a.repeats,
        "host": socket.gethostname(), "platform": platform.platform(),
        "wall_s": time.time() - t_start,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    M.write_profile_manifest(a.out, rows, meta)
    pct = M.latency_percentiles([{"latency_s": float(r["latency_s"])} for r in rows])
    print("\nwrote %s (%d rows, %d failures) in %.0fs"
          % (a.out, len(rows), failures, meta["wall_s"]))
    print("latency percentiles (s): %s"
          % json.dumps({k: round(v, 5) for k, v in pct.items()}))
    for regime in M.RATIO_REGIMES:
        _f, _s, cuts = M.build_pools(
            [{"sample_id": r["sample_id"], "latency_s": float(r["latency_s"])}
             for r in rows], regime)
        print("  %-9s fast=%d slow=%d observed ratio=%s"
              % (regime, cuts["fast_pool_size"], cuts["slow_pool_size"],
                 round(cuts["observed_cost_ratio"], 2)
                 if cuts.get("observed_cost_ratio") else "n/a"))


if __name__ == "__main__":
    main()
