from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT = ROOT / "outputs"


def ensure_output_dir():
    OUT.mkdir(exist_ok=True)
    return OUT
