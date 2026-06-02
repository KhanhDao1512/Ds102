# Module structure

```text
Lab5/
|-- lab5_solution.py          # Backward-compatible wrapper
|-- scripts/
|   `-- run_lab5.py           # Standard runner script
|-- src/
|   `-- lab5/
|       |-- __init__.py
|       |-- core/
|       |   |-- kmeans.py     # NumPy K-means implementation
|       |   `-- gmm.py        # NumPy GMM + EM implementation
|       |-- datasets/
|       |   `-- data.py       # Generate Gaussian toy datasets
|       |-- tasks/
|       |   |-- experiments.py
|       |   `-- image_background.py
|       `-- utils/
|           |-- paths.py
|           |-- metrics.py
|           |-- visualization.py
|           `-- reporting.py
|-- lab5_nhan_xet.md          # Vietnamese comments for the report
`-- outputs/                  # Generated figures and report
```

Run the whole lab:

```powershell
C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe lab5_solution.py
```

Or run the standard script:

```powershell
C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\run_lab5.py
```

Import examples:

```python
from lab5.core.kmeans import kmeans
from lab5.core.gmm import gmm_em
from lab5.datasets.data import generate_gaussian_data
from lab5.tasks.image_background import run_gmm_background_filter
```
