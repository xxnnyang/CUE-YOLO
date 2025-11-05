#!/usr/bin/env python3
"""
Merge Ultralytics val metrics into one CSV.

Scans: ./summer/runs/val/**/metrics.json
Writes: val_metrics_YYYYMMDD_HHMMSS.csv

Usage (run in the project root):
    python merge_val_metrics.py
or to set a different base dir:
    python merge_val_metrics.py --base ./summer/runs/val

Notes:
- Each row corresponds to one metrics.json.
- Columns are the union of all keys across files, plus:
    run_dir (relative), metrics_path, mtime
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

def find_metrics_files(base: Path) -> List[Path]:
    # Search for metrics.json under base (non-recursive folders under val or deeper)
    return list(base.rglob("metrics.json"))

def read_json_safe(p: Path) -> Dict[str, Any]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {p}: {e}", file=sys.stderr)
        return {}

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="./summer/runs/val",
                    help="Base directory to scan (default: ./summer/runs/val)")
    ap.add_argument("--out", type=str, default="",
                    help="Output CSV path (default: val_metrics_YYYYMMDD_HHMMSS.csv in CWD)")
    args = ap.parse_args(argv)

    base = Path(args.base).resolve()
    if not base.exists():
        print(f"[ERROR] Base directory does not exist: {base}", file=sys.stderr)
        return 2

    files = find_metrics_files(base)
    if not files:
        print(f"[INFO] No metrics.json found under: {base}")
        return 0

    # Read all JSONs and collect union of keys
    rows: List[Dict[str, Any]] = []
    all_keys: set[str] = set()
    for f in sorted(files):
        data = read_json_safe(f)
        # augment with metadata
        rel_run_dir = str(f.parent.relative_to(base))
        mtime = datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")
        data["_run_dir"] = rel_run_dir
        data["_metrics_path"] = str(f)
        data["_mtime"] = mtime

        rows.append(data)
        all_keys.update(data.keys())

    # Ensure consistent ordering: metadata first, then metric keys (sorted)
    meta_cols = ["_run_dir", "_metrics_path", "_mtime"]
    metric_cols = sorted(k for k in all_keys if k not in meta_cols)
    header = meta_cols + metric_cols

    # Decide output path
    out_path = Path(args.out) if args.out else Path.cwd() / f"val_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # Write CSV
    with out_path.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=header)
        w.writeheader()
        for r in rows:
            # Fill missing keys with empty string
            w.writerow({k: r.get(k, "") for k in header})

    print(f"[OK] Wrote {len(rows)} rows -> {out_path}")
    print("[COLUMNS]", ", ".join(header))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


# python merge_val_metrics.py --out ./summer/runs/val/GC-DET-new/val_metrics_all.csv --base ./summer/runs/val/GC-DET-new
# python merge_val_metrics.py --out ./summer/runs/test/NEU-DET-flip/val_metrics_all.csv --base ./summer/runs/test/NEU-DET-flip