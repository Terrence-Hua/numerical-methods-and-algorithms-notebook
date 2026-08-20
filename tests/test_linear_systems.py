"""Tests for src/linear_systems.py — validated against numpy reference solutions."""

from __future__ import annotations

import numpy as np
import pytest

from src.linear_systems import (
    cholesky_decompose,
    cholesky_solve,
    conjugate_gradient,
    lu_decompose,
    lu_solve,
    preconditioned_cg,
    qr_decompose,
    qr_solve,
)

RNG = np.random.default_rng(12345)
TOL = 1e-8


def _spd(n: int, seed: int = 0) -> np.ndarray:
    """Return a random symmetric positive-definite matrix of size n."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


# ---------------------------------------------------------------------------
# LU decomposition
# ---------------------------------------------------------------------------


class TestLU:
    def test_factorisation_identity(self):
        A = _spd(5)
        L, U, P = lu_decompose(A)
        assert np.allclose(P @ A, L @ U, atol=1e-12)

    def test_L_lower_triangular(self):
        A = _spd(6)
        L, _, _ = lu_decompose(A)
        assert np.allclose(np.tril(L), L)
        # Unit diagonal
        assert np.allclose(np.diag(L), np.ones(6))

    def test_U_upper_triangular(self):
        A = _spd(6)
        _, U, _ = lu_decompose(A)
        assert np.allclose(np.triu(U), U)

    def test_P_permutation(self):
        A = _spd(6)
        _, _, P = lu_decompose(A)
        # P is orthogonal
        assert np.allclose(P @ P.T, np.eye(6))

    def test_solve_small(self):
        A = _spd(4)
        b = RNG.standard_normal(4)
        L, U, P = lu_decompose(A)
        x = lu_solve(L, U, P, b)
        assert np.linalg.norm(A @ x - b) < TOL

    def test_solve_larger(self):
        A = _spd(20, seed=1)
        b = np.random.default_rng(1).standard_normal(20)
        L, U, P = lu_decompose(A)
        x = lu_solve(L, U, P, b)
        assert np.linalg.norm(A @ x - b) < TOL

    def test_matches_numpy(self):
        A = _spd(8)
        b = RNG.standard_normal(8)
        L, U, P = lu_decompose(A)
        x_lu = lu_solve(L, U, P, b)
        x_np = np.linalg.solve(A, b)
        assert np.allclose(x_lu, x_np, atol=1e-10)

    def test_singular_raises(self):
        A = np.zeros((3, 3))
        with pytest.raises(np.linalg.LinAlgError):
            lu_decompose(A)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            lu_decompose(np.ones((3, 4)))

    def test_known_2x2(self):
        A = np.array([[2.0, 1.0], [6.0, 4.0]])
        b = np.array([3.0, 10.0])
        L, U, P = lu_decompose(A)
        x = lu_solve(L, U, P, b)
        assert np.allclose(x, np.linalg.solve(A, b), atol=1e-12)


# ---------------------------------------------------------------------------
# QR decomposition
# ---------------------------------------------------------------------------


class TestQR:
    def test_factorisation_identity(self):
        A = _spd(5)
        Q, R = qr_decompose(A)
        assert np.allclose(Q @ R, A, atol=1e-12)

    def test_Q_orthogonal(self):
        A = _spd(6)
        Q, _ = qr_decompose(A)
        assert np.allclose(Q.T @ Q, np.eye(6), atol=1e-12)
        assert np.allclose(Q @ Q.T, np.eye(6), atol=1e-12)

    def test_R_upper_triangular(self):
        A = _spd(5)
        _, R = qr_decompose(A)
        assert np.allclose(np.triu(R), R, atol=1e-12)

    def test_solve_matches_numpy(self):
        A = _spd(8)
        b = np.random.default_rng(2).standard_normal(8)
        Q, R = qr_decompose(A)
        x = qr_solve(Q, R, b)
        x_np = np.linalg.solve(A, b)
        assert np.allclose(x, x_np, atol=1e-10)

    def test_solve_residual(self):
        A = _spd(12, seed=3)
        b = np.random.default_rng(3).standard_normal(12)
        Q, R = qr_decompose(A)
        x = qr_solve(Q, R, b)
        assert np.linalg.norm(A @ x - b) < TOL

    def test_non_square_tall(self):
        rng = np.random.default_rng(99)
        A = rng.standard_normal((8, 4))
        Q, R = qr_decompose(A)
        assert np.allclose(Q @ R, A, atol=1e-12)
        assert Q.shape == (8, 8)
        assert R.shape == (8, 4)

    def test_short_raises(self):
        with pytest.raises(ValueError, match="m >= n"):
            qr_decompose(np.ones((3, 5)))


# ---------------------------------------------------------------------------
# Conjugate Gradient
# ---------------------------------------------------------------------------


class TestCG:
    def test_solve_spd(self):
        A = _spd(10)
        b = np.random.default_rng(4).standard_normal(10)
        x, residuals = conjugate_gradient(A, b, tol=1e-12)
        assert np.linalg.norm(A @ x - b) < 1e-10

    def test_converges_in_n_steps(self):
        n = 8
        A = _spd(n)
        b = np.random.default_rng(5).standard_normal(n)
        _, residuals = conjugate_gradient(A, b, tol=1e-14)
        assert len(residuals) <= n + 2

    def test_residuals_decreasing(self):
        A = _spd(10)
        b = np.random.default_rng(6).standard_normal(10)
        _, residuals = conjugate_gradient(A, b)
        assert residuals[-1] < residuals[0]

    def test_matches_numpy(self):
        A = _spd(15, seed=7)
        b = np.random.default_rng(7).standard_normal(15)
        x, _ = conjugate_gradient(A, b, tol=1e-12)
        x_np = np.linalg.solve(A, b)
        assert np.allclose(x, x_np, atol=1e-9)

    def test_custom_x0(self):
        A = _spd(6)
        b = np.ones(6)
        x_np = np.linalg.solve(A, b)
        x0 = x_np + 0.01 * np.ones(6)
        x, _ = conjugate_gradient(A, b, x0=x0, tol=1e-12)
        assert np.allclose(x, x_np, atol=1e-9)


# ---------------------------------------------------------------------------
# Preconditioned CG
# ---------------------------------------------------------------------------


class TestPCG:
    def test_jacobi_preconditioner(self):
        A = _spd(10, seed=8)
        b = np.random.default_rng(8).standard_normal(10)
        M_inv = np.diag(1.0 / np.diag(A))
        x, residuals = preconditioned_cg(A, b, M_inv=M_inv, tol=1e-12)
        assert np.linalg.norm(A @ x - b) < 1e-10

    def test_no_preconditioner_matches_cg(self):
        A = _spd(8, seed=9)
        b = np.random.default_rng(9).standard_normal(8)
        x_cg, _ = conjugate_gradient(A, b, tol=1e-12)
        x_pcg, _ = preconditioned_cg(A, b, tol=1e-12)
        assert np.allclose(x_cg, x_pcg, atol=1e-9)


# ---------------------------------------------------------------------------
# Cholesky decomposition
# ---------------------------------------------------------------------------


class TestCholesky:
    def test_factorisation(self):
        A = _spd(6)
        L = cholesky_decompose(A)
        assert np.allclose(L @ L.T, A, atol=1e-12)

    def test_L_lower_triangular(self):
        A = _spd(5)
        L = cholesky_decompose(A)
        assert np.allclose(np.tril(L), L)

    def test_solve_matches_numpy(self):
        A = _spd(8, seed=10)
        b = np.random.default_rng(10).standard_normal(8)
        L = cholesky_decompose(A)
        x = cholesky_solve(L, b)
        x_np = np.linalg.solve(A, b)
        assert np.allclose(x, x_np, atol=1e-10)

    def test_not_positive_definite_raises(self):
        A = -np.eye(4)
        with pytest.raises(np.linalg.LinAlgError):
            cholesky_decompose(A)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            cholesky_decompose(np.ones((3, 4)))

    def test_known_2x2(self):
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        L = cholesky_decompose(A)
        assert np.allclose(L @ L.T, A, atol=1e-14)
        # L[0,0] = 2, L[1,0] = 1, L[1,1] = sqrt(2)
        assert abs(L[0, 0] - 2.0) < 1e-14
        assert abs(L[1, 0] - 1.0) < 1e-14
        assert abs(L[1, 1] - np.sqrt(2)) < 1e-14
