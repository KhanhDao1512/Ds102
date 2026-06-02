from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab5.tasks.experiments import run_gmm_toy, run_kmeans_assignments
from lab5.tasks.image_background import run_gmm_background_filter
from lab5.utils.reporting import write_report


def main():
    kmeans_results = run_kmeans_assignments()
    gmm_result = run_gmm_toy()
    background_result = run_gmm_background_filter()
    report = write_report(kmeans_results, gmm_result, background_result)
    print(report)


if __name__ == "__main__":
    main()

