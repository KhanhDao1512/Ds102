import numpy as np

from lab5.utils.paths import OUT, ensure_output_dir


def write_report(kmeans_results, gmm_result, background_result):
    ensure_output_dir()
    lines = []
    lines.append("LAB 5 SOLUTION SUMMARY")
    lines.append("")
    lines.append("K-MEANS CLUSTERING")
    for result in kmeans_results:
        lines.append(f"- {result['name']}")
        lines.append(f"  points: {result['n_points']}")
        lines.append(f"  iterations: {result['iterations']}")
        lines.append(f"  final objective: {result['objective']:.4f}")
        lines.append(f"  cluster purity against generated labels: {result['purity']:.4f}")
        lines.append(f"  centers:\n{np.array2string(result['centers'], precision=4)}")
        lines.append(f"  comment: {result['comment']}")
        lines.append(f"  figure: outputs/{result['image']}")
        lines.append("")

    lines.append("GAUSSIAN MIXTURE MODEL")
    lines.append(f"- toy GMM iterations: {gmm_result['iterations']}")
    lines.append(f"- toy GMM final log likelihood: {gmm_result['log_likelihood']:.4f}")
    lines.append(f"- toy GMM purity against generated labels: {gmm_result['purity']:.4f}")
    lines.append(f"- learned weights: {np.array2string(gmm_result['weights'], precision=4)}")
    lines.append(f"- learned means:\n{np.array2string(gmm_result['means'], precision=4)}")
    lines.append(f"- figure: outputs/{gmm_result['image']}")
    lines.append("")

    lines.append("BACKGROUND FILTERING WITH GMM")
    lines.append(f"- input image size: {background_result['image_size']}")
    lines.append(f"- selected background component: {background_result['background_component']}")
    lines.append(f"- foreground pixel ratio: {background_result['foreground_ratio']:.4f}")
    for output in background_result["outputs"]:
        lines.append(f"- output: outputs/{output}")
    lines.append("")
    lines.append(
        "Note: all clustering algorithms above are implemented with NumPy. PIL is used only for reading/writing images and drawing simple figures."
    )

    report = "\n".join(lines)
    (OUT / "lab5_report.txt").write_text(report, encoding="utf-8")
    return report
