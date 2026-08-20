"""Linear system solvers: LU (partial pivoting), QR (Householder), CG, Cholesky."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# LU decomposition with partial pivoting
# ---------------------------------------------------------------------------


def lu_decompose(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor A into P A = L U using partial pivoting.

    Args:
        A: Square matrix of shape (n, n).

    Returns:
        (L, U, P) where:
          - L is unit lower-triangular, shape (n, n)
          - U is upper-triangular, shape (n, n)
          - P is the permutation matrix, shape (n, n)
          satisfying P @ A = L @ U.

    Raises:
        ValueError: If A is not square.
        np.linalg.LinAlgError: If A is singular to working precision.
    """
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError(f"A must be square; got shape {A.shape}")

    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        # Find pivot
        pivot_row = k + np.argmax(np.abs(U[k:, k]))
        if np.abs(U[pivot_row, k]) < 1e-14:
            raise np.linalg.LinAlgError("Matrix is singular to working precision")

        if pivot_row != k:
            U[[k, pivot_row]] = U[[pivot_row, k]]
            P[[k, pivot_row]] = P[[pivot_row, k]]
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        for i in range(k + 1, n):
            if U[k, k] == 0.0:
                continue
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    return L, U, P


def lu_solve(
    L: np.ndarray, U: np.ndarray, P: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Solve A x = b given the LU factorisation P A = L U.

    Uses forward substitution on L y = P b, then back substitution on U x = y.

    Args:
        L: Unit lower-triangular matrix, shape (n, n).
        U: Upper-triangular matrix, shape (n, n).
        P: Permutation matrix, shape (n, n).
        b: Right-hand side vector, shape (n,).

    Returns:
        Solution vector x of shape (n,).
    """
    pb = P @ b
    n = len(pb)

    # Forward substitution: L y = P b
    y = np.zeros(n)
    for i in range(n):
        y[i] = pb[i] - L[i, :i] @ y[:i]

    # Back substitution: U x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]

    return x


# ---------------------------------------------------------------------------
# QR decomposition via Householder reflections
# ---------------------------------------------------------------------------


def qr_decompose(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Factor A into Q R using Householder reflections.

    Args:
        A: Matrix of shape (m, n) with m >= n.

    Returns:
        (Q, R) where Q is orthogonal (m, m) and R is upper-triangular (m, n),
        satisfying A = Q @ R.

    Raises:
        ValueError: If m < n.
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    if m < n:
        raise ValueError(f"Require m >= n; got shape {A.shape}")

    Q = np.eye(m)
    R = A.copy()

    for k in range(n):
        x = R[k:, k].copy()
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-14:
            continue

        # Build Householder vector
        e1 = np.zeros_like(x)
        e1[0] = norm_x if x[0] <= 0 else -norm_x
        v = x - e1
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-14:
            continue
        v = v / v_norm

        # Apply H_k = I - 2 v v^T to R and Q
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)

    return Q, R


def qr_solve(Q: np.ndarray, R: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b given the QR factorisation A = Q R.

    Args:
        Q: Orthogonal matrix (m, m).
        R: Upper-triangular matrix (m, n).
        b: Right-hand side vector (m,).

    Returns:
        Least-squares solution x of shape (n,), exact for square full-rank A.
    """
    n = R.shape[1]
    Qtb = Q.T @ b

    # Back substitution on R[:n, :n] x = Q^T b [:n]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Qtb[i] - R[i, i + 1 : n] @ x[i + 1 : n]) / R[i, i]

    return x


# ---------------------------------------------------------------------------
# Conjugate Gradient
# ---------------------------------------------------------------------------


def conjugate_gradient(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Solve A x = b via the Conjugate Gradient method.

    A must be symmetric positive-definite. Converges in at most n steps in
    exact arithmetic; in practice terminates when the residual norm drops
    below tol.

    Args:
        A: Symmetric positive-definite matrix, shape (n, n).
        b: Right-hand side vector, shape (n,).
        x0: Initial guess; defaults to the zero vector.
        tol: Convergence tolerance on the residual norm ||r||_2.
        max_iter: Maximum iterations; defaults to n.

    Returns:
        (x, residuals) where x is the solution and residuals[i] = ||r_i||_2.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    if max_iter is None:
        max_iter = n

    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    r = b - A @ x
    p = r.copy()
    residuals: list[float] = [float(np.linalg.norm(r))]

    for _ in range(max_iter):
        if residuals[-1] < tol:
            break
        Ap = A @ p
        rr = r @ r
        alpha = rr / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        beta = (r @ r) / rr
        p = r + beta * p
        residuals.append(float(np.linalg.norm(r)))

    return x, residuals


# ---------------------------------------------------------------------------
# Cholesky decomposition (for symmetric positive-definite matrices)
# ---------------------------------------------------------------------------


def cholesky_decompose(A: np.ndarray) -> np.ndarray:
    """Factor a symmetric positive-definite matrix as A = L L^T.

    Uses the outer-product form (Cholesky-Banachiewicz algorithm).

    Args:
        A: Symmetric positive-definite matrix, shape (n, n).

    Returns:
        L: Lower-triangular matrix such that L @ L.T == A.

    Raises:
        ValueError: If A is not square.
        np.linalg.LinAlgError: If A is not positive-definite.
    """
    A = np.array(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError(f"A must be square; got shape {A.shape}")

    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0.0:
                    raise np.linalg.LinAlgError(
                        "Matrix is not positive-definite"
                    )
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]

    return L


def cholesky_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b given the Cholesky factor L where A = L L^T.

    Args:
        L: Lower-triangular Cholesky factor, shape (n, n).
        b: Right-hand side vector, shape (n,).

    Returns:
        Solution vector x of shape (n,).
    """
    n = len(b)

    # Forward substitution: L y = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]

    # Back substitution: L^T x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L[i + 1 :, i] @ x[i + 1 :]) / L[i, i]

    return x
