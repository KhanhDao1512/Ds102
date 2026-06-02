from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs"


def ensure_output_dir():
    OUT.mkdir(exist_ok=True)
    return OUT

