"""Package course source and measured results without weights or raw demonstrations."""

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    files = set()
    for folder in ["src", "scripts", "tests", "configs", "stimuli", "experiments"]:
        for p in (ROOT / folder).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".log"}:
                files.add(p)
    for name in [
        "README.md",
        "CS1998_README.md",
        "CLAUDE.md",
        "Makefile",
        "pyproject.toml",
        "requirements_cs1998.lock.txt",
        ".gitignore",
        ".gitattributes",
    ]:
        p = ROOT / name
        if p.exists():
            files.add(p)
    for p in (ROOT / "paper").iterdir():
        if p.is_file() and (p.name.startswith("CS1998") or p.suffix in {".tex", ".bib", ".md"}):
            files.add(p)
    for p in (ROOT / "figures").rglob("*.pdf"):
        files.add(p)
    for p in (ROOT / "results/cs1998_full").rglob("*"):
        if p.is_file() and p.name != "partial_activations.npz":
            files.add(p)
    files.add(ROOT / "results/checkpoint_screen.json")
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    target = out / "CS1998_source_and_results.zip"
    manifest = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)
    }
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, str(Path("CS1998_project") / p.relative_to(ROOT)))
        z.writestr("CS1998_project/MANIFEST_SHA256.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name, digest in manifest.items():
            assert hashlib.sha256(z.read("CS1998_project/" + name)).hexdigest() == digest
    print(
        f"{target}: {len(files)} files, "
        f"{target.stat().st_size / 1024**2:.1f} MiB; integrity verified"
    )


if __name__ == "__main__":
    build()
