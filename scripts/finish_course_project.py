"""Finish analysis and documents once the resumable inference cache is complete.

Run as a detached local process when inference is long. This does not publish,
submit, or save to external services. Final human/agent review is separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def finish(run: Path) -> None:
    while not (run / "activations.npz").exists():
        time.sleep(30)
    status = {"stage": "analysis", "started_at_unix": time.time()}
    target = run / "completion.json"

    def save():
        target.write_text(json.dumps(status, indent=2))

    save()
    try:
        for seed in range(5):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.analyze_execution_probe",
                    str(run / "activations.npz"),
                    "--seed",
                    str(seed),
                    "--shuffles",
                    "20",
                    "--resamples",
                    "10000",
                ],
                cwd=ROOT,
                check=True,
            )
        status["stage"] = "numerical_audit"
        save()
        subprocess.run(
            [sys.executable, "-m", "scripts.audit_execution_numerics", str(run)],
            cwd=ROOT,
            check=True,
        )
        hashes = {}
        for path in sorted((ROOT / "data/libero/libero_goal").glob("*.hdf5")):
            h = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    h.update(block)
            hashes[path.name] = {"sha256": h.hexdigest(), "bytes": path.stat().st_size}
        (run / "data_sha256.json").write_text(json.dumps(hashes, indent=2))
        status["stage"] = "documents"
        save()
        subprocess.run(
            [sys.executable, "-m", "scripts.write_execution_report", str(run)], cwd=ROOT, check=True
        )
        subprocess.run(
            [sys.executable, "-m", "scripts.write_course_supplement", str(run)],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "scripts.write_course_detailed", str(run)],
            cwd=ROOT,
            check=True,
        )
        status["stage"] = "ready_for_final_review"
        status["finished_at_unix"] = time.time()
        save()
    except Exception as exc:
        status.update(stage="failed", error=str(exc))
        save()
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    finish(p.parse_args().run.resolve())
