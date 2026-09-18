"""Profiling manifests, percentile pools and deterministic dataset mixing.

Torch-free on purpose: everything in here is exercised by scripts/selftest.py
on any machine, including ones without a working torch install.

Vocabulary
----------
profile manifest   one row per candidate sample, with the latency measured by
                   a single plain-PyTorch pass (script 01).
pool               a latency percentile band of the profile manifest. The fast
                   pool and the slow pool are defined by the ratio regime.
run manifest       the mixed dataset actually fed to a run (script 02): exactly
                   N rows drawn from the two pools in the requested proportion,
                   each carrying its ground-truth pool label.
"""

import csv
import hashlib
import json
import os
import random

PROFILE_FIELDS = [
    "sample_id", "path", "bytes", "latency_s", "stage_times_json", "extra_json",
]

RUN_FIELDS = [
    "order", "sample_id", "path", "bytes", "profiled_latency_s",
    "truth_label", "is_sentinel", "sentinel_role",
]

# Percentile cuts, as decided in the experiment design. (lo, hi) inclusive of
# lo, exclusive of hi, except that hi=100 includes the top.
RATIO_REGIMES = {
    # fast = P0-P50, slow = P75-P90  -> roughly 4-6x typical
    "moderate": {"fast": (0.0, 50.0), "slow": (75.0, 90.0)},
    # fast = P0-P50, slow = P97+     -> roughly 15-20x typical
    "extreme": {"fast": (0.0, 50.0), "slow": (97.0, 100.0)},
}


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def write_profile_manifest(path, rows, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PROFILE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PROFILE_FIELDS})
    with open(_meta_path(path), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


def read_profile_manifest(path):
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            row["bytes"] = int(row["bytes"] or 0)
            row["latency_s"] = float(row["latency_s"])
            rows.append(row)
    if not rows:
        raise ValueError("empty profile manifest: %s" % path)
    return rows


def read_profile_meta(path):
    meta_path = _meta_path(path)
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as fh:
        return json.load(fh)


def _meta_path(path):
    return os.path.splitext(path)[0] + ".meta.json"


def write_run_manifest(path, rows, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in RUN_FIELDS})
    with open(_meta_path(path), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


def read_run_manifest(path):
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            row["order"] = int(row["order"])
            row["bytes"] = int(row["bytes"] or 0)
            row["profiled_latency_s"] = float(row["profiled_latency_s"])
            row["is_sentinel"] = row["is_sentinel"] in ("1", "True", "true")
            rows.append(row)
    rows.sort(key=lambda r: r["order"])
    return rows


# ---------------------------------------------------------------------------
# pools
# ---------------------------------------------------------------------------

def _percentile(sorted_values, q):
    """Linear-interpolation percentile, stdlib only (no numpy dependency)."""
    if not sorted_values:
        raise ValueError("empty sequence")
    if q <= 0:
        return sorted_values[0]
    if q >= 100:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def latency_percentiles(rows, qs=(0, 25, 50, 75, 90, 95, 97, 99, 100)):
    values = sorted(r["latency_s"] for r in rows)
    return {("p%g" % q): _percentile(values, q) for q in qs}


def build_pools(rows, regime):
    """Split the profile manifest into a fast pool and a slow pool.

    Returns (fast_rows, slow_rows, cuts). Rows are ordered by latency so the
    selection is stable regardless of the manifest's original order.
    """
    if regime not in RATIO_REGIMES:
        raise ValueError("unknown ratio regime %r; expected one of %s"
                         % (regime, sorted(RATIO_REGIMES)))
    cutspec = RATIO_REGIMES[regime]
    ordered = sorted(rows, key=lambda r: (r["latency_s"], r["sample_id"]))
    values = [r["latency_s"] for r in ordered]

    def band(lo_q, hi_q):
        lo = _percentile(values, lo_q)
        hi = _percentile(values, hi_q)
        if hi_q >= 100:
            return [r for r in ordered if r["latency_s"] >= lo], (lo, float("inf"))
        return [r for r in ordered if lo <= r["latency_s"] < hi], (lo, hi)

    fast_rows, fast_cut = band(*cutspec["fast"])
    slow_rows, slow_cut = band(*cutspec["slow"])
    cuts = {
        "regime": regime,
        "fast_percentiles": list(cutspec["fast"]),
        "slow_percentiles": list(cutspec["slow"]),
        "fast_latency_range_s": [fast_cut[0], fast_cut[1]],
        "slow_latency_range_s": [slow_cut[0], slow_cut[1]],
        "fast_pool_size": len(fast_rows),
        "slow_pool_size": len(slow_rows),
    }
    if fast_rows and slow_rows:
        fast_med = _percentile([r["latency_s"] for r in fast_rows], 50)
        slow_med = _percentile([r["latency_s"] for r in slow_rows], 50)
        cuts["observed_cost_ratio"] = (slow_med / fast_med) if fast_med > 0 else None
    return fast_rows, slow_rows, cuts


def choose_sentinels(fast_rows, slow_rows):
    """Pick one representative sample from each pool.

    The median-latency member of each pool, which is deterministic and does not
    depend on a seed, so re-running the extractor cannot silently move it.

    Note a subtlety the design brief glosses over: the fast pool is P0-P50 in
    both regimes, so the fast sentinel is shared. The slow pools (P75-P90 and
    P97+) are disjoint, so the slow sentinel is necessarily per-regime. Compare
    slow sentinels within a regime, never across regimes.
    """
    def median_member(rows):
        if not rows:
            return None
        ordered = sorted(rows, key=lambda r: (r["latency_s"], r["sample_id"]))
        return ordered[len(ordered) // 2]

    return {"fast": median_member(fast_rows), "slow": median_member(slow_rows)}


# ---------------------------------------------------------------------------
# mixing
# ---------------------------------------------------------------------------

def mix_dataset(fast_rows, slow_rows, n_samples, slow_fraction, seed,
                sentinels=None, allow_replacement=True):
    """Deterministically draw `n_samples` rows in the requested proportion.

    Every returned row carries `truth_label`, which is the pool it was drawn
    from. That label is the ground truth against which MinatoLoader's runtime
    fast/slow classification is scored.
    """
    if not 0.0 <= slow_fraction <= 1.0:
        raise ValueError("slow_fraction must be in [0, 1]")
    n_slow = int(round(n_samples * slow_fraction))
    n_fast = n_samples - n_slow
    rng = random.Random(seed)
    warnings = []

    def draw(pool, k, label):
        if k <= 0:
            return []
        if not pool:
            raise ValueError("%s pool is empty; profile a larger candidate set"
                             % label)
        if k <= len(pool):
            picked = rng.sample(pool, k)
        elif allow_replacement:
            warnings.append(
                "%s pool has %d rows but %d were requested; sampling with "
                "replacement, so per-sample rows repeat within the epoch"
                % (label, len(pool), k))
            picked = [rng.choice(pool) for _ in range(k)]
        else:
            raise ValueError("%s pool has %d rows, need %d" % (label, len(pool), k))
        return [dict(r, truth_label=label) for r in picked]

    chosen = draw(fast_rows, n_fast, "fast") + draw(slow_rows, n_slow, "slow")

    # Force the sentinels in, replacing a same-label member so the requested
    # proportion is preserved exactly.
    sentinel_ids = {}
    if sentinels:
        for role in ("fast", "slow"):
            sent = sentinels.get(role)
            if not sent:
                continue
            already = [r for r in chosen if r["sample_id"] == sent["sample_id"]]
            if already:
                for r in already:
                    r["is_sentinel"] = True
                    r["sentinel_role"] = role
                sentinel_ids[role] = sent["sample_id"]
                continue
            victims = [i for i, r in enumerate(chosen)
                       if r["truth_label"] == role and not r.get("is_sentinel")]
            if not victims:
                warnings.append("no room to inject the %s sentinel at "
                                "slow_fraction=%.3f" % (role, slow_fraction))
                continue
            chosen[victims[0]] = dict(sent, truth_label=role, is_sentinel=True,
                                      sentinel_role=role)
            sentinel_ids[role] = sent["sample_id"]

    rng.shuffle(chosen)

    out = []
    for i, row in enumerate(chosen):
        out.append({
            "order": i,
            "sample_id": row["sample_id"],
            "path": row["path"],
            "bytes": row.get("bytes", 0),
            "profiled_latency_s": row["latency_s"],
            "truth_label": row["truth_label"],
            "is_sentinel": "1" if row.get("is_sentinel") else "0",
            "sentinel_role": row.get("sentinel_role", ""),
        })

    meta = {
        "n_samples": n_samples,
        "n_fast": n_fast,
        "n_slow": n_slow,
        "slow_fraction_requested": slow_fraction,
        "slow_fraction_actual": (n_slow / float(n_samples)) if n_samples else 0.0,
        "seed": seed,
        "sentinel_ids": sentinel_ids,
        "warnings": warnings,
    }
    return out, meta


def manifest_fingerprint(rows):
    """Stable hash of a run manifest, to prove two runs used the same data."""
    h = hashlib.sha256()
    for row in rows:
        h.update(("%s|%s|%s\n" % (row["order"], row["sample_id"],
                                  row["truth_label"])).encode())
    return h.hexdigest()[:16]
