# Numerical methods and algorithms notebook

Implementations of core numerical methods and classic algorithms from scratch in Python, with Jupyter notebooks covering derivations, code, and error analysis.

## Topics

| Notebook | Module | Methods |
|---|---|---|
| [Root finding](notebooks/01_root_finding.ipynb) | `src/root_finding.py` | Bisection, Newton-Raphson, Secant, Brent |
| [Linear systems](notebooks/02_linear_systems.ipynb) | `src/linear_systems.py` | LU (pivoted), QR (Householder), Conjugate Gradient |
| [Numerical integration](notebooks/03_integration.ipynb) | `src/integration.py` | Trapezoid, Simpson, Gauss-Legendre |
| [DP and graphs](notebooks/04_dp_and_graphs.ipynb) | `src/dp_algorithms.py`, `src/graph_algorithms.py` | LCS, Knapsack, Edit Distance, Dijkstra, Bellman-Ford, Floyd-Warshall |

## Run

```bash
pip install -r requirements.txt
jupyter lab
```

## Test

```bash
pytest tests/
```

All solvers are validated against scipy or closed-form analytic results.
