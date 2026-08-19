"""Tests for src/root_finding.py — validated against closed-form roots."""

from __future__ import annotations

import math

import pytest

from src.root_finding import bisection, brent, newton, secant

TOL = 1e-8


# ---------------------------------------------------------------------------
# Bisection
# ---------------------------------------------------------------------------


class TestBisection:
    def test_finds_pi_via_sin(self):
        root, iters, errors = bisection(math.sin, 3.0, 4.0, tol=TOL)
        assert abs(root - math.pi) < TOL
        assert iters > 0
        assert errors[-1] < TOL

    def test_finds_sqrt2(self):
        root, _, _ = bisection(lambda x: x**2 - 2, 1.0, 2.0, tol=TOL)
        assert abs(root - math.sqrt(2)) < TOL

    def test_cubic_root(self):
        # x^3 - x - 2 = 0, root near 1.5213797068
        # bisection guarantees interval width < TOL, not |f(root)| < TOL
        true_root = 1.5213797068497547
        root, _, _ = bisection(lambda x: x**3 - x - 2, 1.0, 2.0, tol=TOL)
        assert abs(root - true_root) < TOL

    def test_wrong_signs_raises(self):
        with pytest.raises(ValueError, match="opposite signs"):
            bisection(lambda x: x**2 + 1, 0.0, 1.0)

    def test_exact_midpoint_terminates(self):
        # f(0) = -1, f(2) = 1, root at x=1 exactly
        root, iters, _ = bisection(lambda x: x - 1, 0.0, 2.0, tol=1e-14)
        assert abs(root - 1.0) < 1e-12

    def test_convergence_monotone(self):
        _, _, errors = bisection(math.sin, 3.0, 4.0)
        for e1, e2 in zip(errors, errors[1:]):
            assert e2 <= e1 + 1e-15


# ---------------------------------------------------------------------------
# Newton-Raphson
# ---------------------------------------------------------------------------


class TestNewton:
    def test_sqrt2(self):
        root, iters, _ = newton(lambda x: x**2 - 2, lambda x: 2 * x, 1.5, tol=TOL)
        assert abs(root - math.sqrt(2)) < TOL
        assert iters < 20

    def test_pi_via_sin(self):
        root, _, errors = newton(math.sin, math.cos, 3.0, tol=TOL)
        assert abs(root - math.pi) < TOL
        assert errors[-1] < TOL

    def test_cube_root(self):
        root, _, _ = newton(lambda x: x**3 - 2, lambda x: 3 * x**2, 1.0, tol=TOL)
        assert abs(root - 2 ** (1 / 3)) < TOL

    def test_quadratic_convergence(self):
        # Newton converges quadratically: each step reduces |f(x)| by at least
        # a factor of 100 once we're close to the root.
        _, _, errors = newton(lambda x: x**2 - 2, lambda x: 2 * x, 1.5, tol=1e-14)
        fast_steps = 0
        for i in range(len(errors) - 1):
            if errors[i] > 0 and errors[i + 1] > 0:
                if errors[i] < 0.1 and errors[i + 1] < errors[i] / 100:
                    fast_steps += 1
        assert fast_steps >= 2

    def test_zero_derivative_raises(self):
        with pytest.raises(ZeroDivisionError):
            newton(lambda x: x**2, lambda x: 0.0, 1.0)


# ---------------------------------------------------------------------------
# Secant
# ---------------------------------------------------------------------------


class TestSecant:
    def test_sqrt2(self):
        root, _, _ = secant(lambda x: x**2 - 2, 1.0, 2.0, tol=TOL)
        assert abs(root - math.sqrt(2)) < TOL

    def test_pi_via_sin(self):
        root, _, _ = secant(math.sin, 3.0, 3.5, tol=TOL)
        assert abs(root - math.pi) < TOL

    def test_exponential(self):
        # e^x - 3 = 0, root = ln(3)
        root, _, _ = secant(lambda x: math.exp(x) - 3, 1.0, 1.5, tol=TOL)
        assert abs(root - math.log(3)) < TOL

    def test_returns_three_tuple(self):
        result = secant(lambda x: x - 1, 0.0, 2.0)
        assert len(result) == 3
        root, iters, errors = result
        assert isinstance(iters, int)
        assert isinstance(errors, list)


# ---------------------------------------------------------------------------
# Brent
# ---------------------------------------------------------------------------


class TestBrent:
    def test_finds_pi_via_sin(self):
        root, iters, _ = brent(math.sin, 3.0, 4.0, tol=TOL)
        assert abs(root - math.pi) < TOL
        assert iters > 0

    def test_cubic(self):
        # x^3 - x - 2, root at ~1.5214
        root, _, _ = brent(lambda x: x**3 - x - 2, 1.0, 2.0, tol=TOL)
        assert abs(root**3 - root - 2) < TOL

    def test_sqrt2(self):
        root, _, _ = brent(lambda x: x**2 - 2, 1.0, 2.0, tol=TOL)
        assert abs(root - math.sqrt(2)) < TOL

    def test_wrong_signs_raises(self):
        with pytest.raises(ValueError, match="opposite signs"):
            brent(lambda x: x**2 + 1, 0.0, 1.0)

    def test_log_equation(self):
        # x*ln(x) - 1 = 0, root at ~1.7632
        import math as m

        root, _, _ = brent(lambda x: x * m.log(x) - 1, 1.0, 3.0, tol=TOL)
        assert abs(root * math.log(root) - 1) < TOL

    def test_faster_than_bisection(self):
        # Brent should converge in fewer iterations than bisection for smooth f
        _, brent_iters, _ = brent(lambda x: x**5 - x - 1, 1.0, 2.0, tol=TOL)
        _, bis_iters, _ = bisection(lambda x: x**5 - x - 1, 1.0, 2.0, tol=TOL)
        assert brent_iters <= bis_iters
