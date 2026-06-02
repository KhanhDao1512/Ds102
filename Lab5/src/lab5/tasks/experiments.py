from lab5.core.gmm import gmm_em
from lab5.core.kmeans import kmeans
from lab5.datasets.data import generate_gaussian_data
from lab5.utils.metrics import cluster_purity
from lab5.utils.paths import OUT, ensure_output_dir
from lab5.utils.visualization import draw_scatter


def run_kmeans_assignments():
    ensure_output_dir()
    sigma_i = [[1, 0], [0, 1]]
    sigma_wide = [[10, 0], [0, 1]]

    assignments = [
        (
            "kmeans_assignment_1",
            [
                (200, [2, 2], sigma_i),
                (200, [8, 3], sigma_i),
                (200, [3, 6], sigma_i),
            ],
            "Random centroid initialization can lead to different local minima. "
            "Running K-means several times and keeping the smallest objective is usually better.",
        ),
        (
            "kmeans_assignment_2",
            [
                (1200, [2, 2], sigma_i),
                (200, [8, 3], sigma_i),
                (1000, [3, 6], sigma_i),
            ],
            "Unequal cluster sizes make K-means biased toward large clusters. "
            "Small clusters can be absorbed or their centroids can move toward denser regions.",
        ),
        (
            "kmeans_assignment_3",
            [
                (200, [2, 2], sigma_i),
                (200, [8, 3], sigma_i),
                (200, [3, 6], sigma_wide),
            ],
            "The third Gaussian is stretched along the x-axis. K-means assumes spherical clusters, "
            "so an elongated cluster is harder to represent with one centroid.",
        ),
    ]

    summaries = []
    for index, (name, specs, comment) in enumerate(assignments, start=1):
        x, y = generate_gaussian_data(specs, seed=42 + index)
        labels, centers, history = kmeans(x, 3, seed=7)
        purity = cluster_purity(y, labels, 3)
        image_path = OUT / f"{name}.png"
        draw_scatter(x, labels, centers, image_path, f"{name}: K-means result")
        summaries.append(
            {
                "name": name,
                "n_points": len(x),
                "iterations": len(history),
                "objective": history[-1],
                "purity": purity,
                "centers": centers,
                "comment": comment,
                "image": image_path.name,
            }
        )
    return summaries


def run_gmm_toy():
    ensure_output_dir()
    sigma_i = [[1, 0], [0, 1]]
    specs = [
        (200, [2, 2], sigma_i),
        (200, [8, 3], sigma_i),
        (200, [3, 6], sigma_i),
    ]
    x, y = generate_gaussian_data(specs, seed=123)
    model = gmm_em(x, 3, seed=8)
    labels = model["labels"]
    purity = cluster_purity(y, labels, 3)
    draw_scatter(x, labels, model["means"], OUT / "gmm_toy_assignment.png", "gmm_toy_assignment: GMM result")
    return {
        "iterations": len(model["log_likelihoods"]),
        "log_likelihood": model["log_likelihoods"][-1],
        "purity": purity,
        "weights": model["weights"],
        "means": model["means"],
        "image": "gmm_toy_assignment.png",
    }

