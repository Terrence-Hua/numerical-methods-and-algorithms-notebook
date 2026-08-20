"""Tests for src/integration.py — validated against analytic results."""

from __future__ import annotations

import math

import pytest

from src.integration import gauss_legendre, simpson, trapezoid

TOL_TRAP = 1e-4   # trapezoid with n=1000 is O(h^2)
TOL_SIMP = 1e-7   # Simpson with n=100, O(h^4); transcendentals over [0,pi] reach ~1e-8
TOL_GL = 2e-7     # GL order 5 on transcendentals over larger intervals


# ---------------------------------------------------------------------------
# Helper integrals with closed-form answers
# ---------------------------------------------------------------------------

CASES = [
    # (f, a, b, exact)
    (lambda x: x**2, 0.0, 1.0, 1 / 3),
    (lambda x: math.sin(x), 0.0, math.pi, 2.0),
    (lambda x: math.exp(x), 0.0, 1.0, math.e - 1),
    (lambda x: 1.0 / (1.0 + x**2), 0.0, 1.0, math.pi / 4),
    (lambda x: x**3, 0.0, 2.0, 4.0),
]


# ---------------------------------------------------------------------------
# Trapezoid rule
# ---------------------------------------------------------------------------


class TestTrapezoid:
    @pytest.mark.parametrize("f,a,b,exact", CASES)
    def test_accuracy(self, f, a, b, exact):
        result, _ = trapezoid(f, a, b, n=1000)
        assert abs(result - exact) < TOL_TRAP

    def test_exact_for_linear(self):
        # Trapezoid is exact for degree-1 polynomials
        result, _ = trapezoid(lambda x: 3 * x + 1, 0.0, 4.0, n=4)
        assert abs(result - 28.0) < 1e-14

    def test_error_decreases_with_n(self):
        exact = 2.0
        r_coarse, _ = trapezoid(math.sin, 0.0, math.pi, n=10)
        r_fine, _ = trapezoid(math.sin, 0.0, math.pi, n=100)
        assert abs(r_fine - exact) < abs(r_coarse - exact)

    def test_richardson_error_estimate(self):
        # Error estimate should bracket the true error for smooth f
        result, err_est = trapezoid(math.exp, 0.0, 1.0, n=100)
        true_err = abs(result - (math.e - 1))
        assert err_est > 0
        # Estimate should be in the right ballpark (within 2 orders of magnitude)
        assert err_est < 100 * true_err + 1e-15

    def test_n1_works(self):
        result, _ = trapezoid(lambda x: 1.0, 0.0, 1.0, n=1)
        assert abs(result - 1.0) < 1e-14

    def test_n_lt_1_raises(self):
        with pytest.raises(ValueError, match="n must be"):
            trapezoid(lambda x: x, 0.0, 1.0, n=0)

    def test_a_ge_b_raises(self):
        with pytest.raises(ValueError, match="Require a < b"):
            trapezoid(lambda x: x, 1.0, 0.0)


# ---------------------------------------------------------------------------
# Simpson's rule
# ---------------------------------------------------------------------------


class TestSimpson:
    @pytest.mark.parametrize("f,a,b,exact", CASES)
    def test_accuracy(self, f, a, b, exact):
        result, _ = simpson(f, a, b, n=100)
        assert abs(result - exact) < TOL_SIMP

    def test_exact_for_cubic(self):
        # Simpson is exact for degree <= 3
        result, _ = simpson(lambda x: x**3, 0.0, 1.0, n=2)
        assert abs(result - 0.25) < 1e-14

    def test_beats_trapezoid_same_n(self):
        # For sin on [0, pi], Simpson with n=10 should beat trapezoid with n=10
        exact = 2.0
        r_simp, _ = simpson(math.sin, 0.0, math.pi, n=10)
        r_trap, _ = trapezoid(math.sin, 0.0, math.pi, n=10)
        assert abs(r_simp - exact) < abs(r_trap - exact)

    def test_error_decreases_as_h4(self):
        exact = math.e - 1
        r10, _ = simpson(math.exp, 0.0, 1.0, n=10)
        r100, _ = simpson(math.exp, 0.0, 1.0, n=100)
        err10 = abs(r10 - exact)
        err100 = abs(r100 - exact)
        # Ratio should be ~10^4 (h goes from 0.1 to 0.01)
        assert err10 / err100 > 100

    def test_odd_n_raises(self):
        with pytest.raises(ValueError, match="even"):
            simpson(lambda x: x, 0.0, 1.0, n=3)

    def test_n_lt_2_raises(self):
        with pytest.raises(ValueError, match="n must be"):
            simpson(lambda x: x, 0.0, 1.0, n=1)

    def test_a_ge_b_raises(self):
        with pytest.raises(ValueError, match="Require a < b"):
            simpson(lambda x: x, 2.0, 1.0)


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature
# ---------------------------------------------------------------------------


class TestGaussLegendre:
    @pytest.mark.parametrize("f,a,b,exact", CASES)
    def test_accuracy_order5(self, f, a, b, exact):
        result, _ = gauss_legendre(f, a, b, order=5)
        assert abs(result - exact) < TOL_GL

    def test_exact_for_degree9_polynomial(self):
        # order=5 GL is exact for polynomials of degree <= 9
        result, _ = gauss_legendre(lambda x: x**9, 0.0, 1.0, order=5)
        assert abs(result - 0.1) < 1e-14

    def test_higher_order_more_accurate(self):
        exact = math.e - 1
        r3, _ = gauss_legendre(math.exp, 0.0, 1.0, order=3)
        r7, _ = gauss_legendre(math.exp, 0.0, 1.0, order=7)
        assert abs(r7 - exact) <= abs(r3 - exact)

    def test_computed_nodes_high_order(self):
        # order=6 uses the Golub-Welsch path; order=10 gives ~1e-15 on smooth f
        result, _ = gauss_legendre(math.sin, 0.0, math.pi, order=10)
        assert abs(result - 2.0) < 1e-10

    def test_order_lt_2_raises(self):
        with pytest.raises(ValueError, match="order must be"):
            gauss_legendre(lambda x: x, 0.0, 1.0, order=1)

    def test_a_ge_b_raises(self):
        with pytest.raises(ValueError, match="Require a < b"):
            gauss_legendre(lambda x: x, 1.0, 0.0)

    def test_error_estimate_reasonable(self):
        result, err = gauss_legendre(math.exp, 0.0, 1.0, order=5)
        true_err = abs(result - (math.e - 1))
        # Error estimate should be non-negative and reasonable
        assert err >= 0
        assert err < 1.0
