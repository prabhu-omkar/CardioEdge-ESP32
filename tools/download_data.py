"""Downloads the five PhysioNet databases into data/raw. About 1.9 GB.

Safe to re-run: finished files are skipped, so a dropped connection only costs a retry.

    python tools/download_data.py
    python tools/download_data.py --only nstdb
"""
import argparse
import csv
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://physionet.org/files"


def get(url, retries=6):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except OSError:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {url}")


def fetch(job):
    url, path = job
    if path.exists() and path.stat().st_size > 0:
        return
    data = get(url)
    if data is None:
        return  # Two AF database records have no signal file
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def records(db, version):
    return get(f"{BASE}/{db}/{version}/RECORDS").decode().split()


def jobs_for(name, raw):
    if name == "mitdb":
        return [(f"{BASE}/mitdb/1.0.0/{r}.{e}", raw / "mitdb" / f"{r}.{e}") for r in records("mitdb", "1.0.0") for e in ("hea", "dat", "atr")]
    if name == "nstdb":
        return [(f"{BASE}/nstdb/1.0.0/{r}.{e}", raw / "nst" / f"{r}.{e}") for r in ("em", "ma", "bw") for e in ("hea", "dat")]
    if name == "afdb":
        return [(f"{BASE}/afdb/1.0.0/{r}.{e}", raw / "afdb" / f"{r}.{e}") for r in records("afdb", "1.0.0") for e in ("hea", "dat", "atr", "qrs")]
    if name == "incartdb":
        return [(f"{BASE}/incartdb/1.0.0/{r}.{e}", raw / "incart" / f"{r}.{e}") for r in records("incartdb", "1.0.0") for e in ("hea", "dat", "atr")]
    if name == "ptbxl":
        with open(ROOT / "ml" / "ptbxl_selection.csv") as f:
            names = [row["filename_hr"] for row in csv.DictReader(f)]
        return [(f"{BASE}/ptb-xl/1.0.3/{r}.{e}", raw / "ptbxl" / f"{r}.{e}") for r in names for e in ("hea", "dat")]
    raise SystemExit(f"Unknown database {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="mitdb, nstdb, afdb, incartdb or ptbxl")
    ap.add_argument("--dest", default=str(ROOT / "data" / "raw"))
    a = ap.parse_args()
    raw = Path(a.dest)
    for name in [a.only] if a.only else ["mitdb", "nstdb", "afdb", "incartdb", "ptbxl"]:
        jobs = jobs_for(name, raw)
        print(f"{name}: {len(jobs)} files", flush=True)
        with ThreadPoolExecutor(8) as pool:
            for i, _ in enumerate(pool.map(fetch, jobs), 1):
                if i % 100 == 0:
                    print(f"  {i}/{len(jobs)}", flush=True)
    print("Done")


if __name__ == "__main__":
    sys.exit(main())
