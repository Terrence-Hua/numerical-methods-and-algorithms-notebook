"""Numerical integration: trapezoid rule, Simpson's rule, Gauss-Legendre quadrature."""

from __future__ import annotations

from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Trapezoid rule
# ---------------------------------------------------------------------------


def trapezoid(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 100,
) -> tuple[float, float]:
    """Approximate the integral of f from a to b using the composite trapezoid rule.

    Uses n equal-width subintervals. Error is O(h^2) where h = (b-a)/n.

    Args:
        f: Integrand, callable on [a, b].
        a: Left endpoint.
        b: Right endpoint.
        n: Number of subintervals (must be >= 1).

    Returns:
        (result, error_estimate) where error_estimate uses Richardson
        extrapolation: |I(n) - I(n/2)| / 3 (requires n even; falls back
        to half the step-size contribution otherwise).

    Raises:
        ValueError: If n < 1 or a >= b.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if a >= b:
        raise ValueError(f"Require a < b; got a={a}, b={b}")

    h = (b - a) / n
    xs = np.linspace(a, b, n + 1)
    ys = np.array([f(x) for x in xs])
    result = h * (0.5 * ys[0] + np.sum(ys[1:-1]) + 0.5 * ys[-1])

    # Richardson error estimate using n//2 panels
    if n >= 2 and n % 2 == 0:
        n2 = n // 2
        h2 = (b - a) / n2
        xs2 = np.linspace(a, b, n2 + 1)
        ys2 = np.array([f(x) for x in xs2])
        result2 = h2 * (0.5 * ys2[0] + np.sum(ys2[1:-1]) + 0.5 * ys2[-1])
        error_estimate = abs(result - result2) / 3.0
    else:
        error_estimate = 0.5 * h**2 * abs(b - a)

    return float(result), float(error_estimate)


# ---------------------------------------------------------------------------
# Simpson's rule
# ---------------------------------------------------------------------------


def simpson(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 100,
) -> tuple[float, float]:
    """Approximate the integral of f from a to b using composite Simpson's 1/3 rule.

    n must be even. Error is O(h^4) where h = (b-a)/n, one order better
    than the trapezoid rule.

    Args:
        f: Integrand, callable on [a, b].
        a: Left endpoint.
        b: Right endpoint.
        n: Number of subintervals (must be even, >= 2).

    Returns:
        (result, error_estimate) using Richardson extrapolation
        |I(n) - I(n/2)| / 15.

    Raises:
        ValueError: If n < 2, n is odd, or a >= b.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2; got {n}")
    if n % 2 != 0:
        raise ValueError(f"n must be even; got {n}")
    if a >= b:
        raise ValueError(f"Require a < b; got a={a}, b={b}")

    h = (b - a) / n
    xs = np.linspace(a, b, n + 1)
    ys = np.array([f(x) for x in xs])

    # Composite Simpson: (h/3) [y0 + 4y1 + 2y2 + 4y3 + ... + 4y_{n-1} + yn]
    coeffs = np.ones(n + 1)
    coeffs[1:-1:2] = 4  # odd indices
    coeffs[2:-2:2] = 2  # even interior indices
    result = (h / 3.0) * float(coeffs @ ys)

    # Richardson error estimate using n//2 panels
    n2 = n // 2
    if n2 >= 2 and n2 % 2 == 0:
        h2 = (b - a) / n2
        xs2 = np.linspace(a, b, n2 + 1)
        ys2 = np.array([f(x) for x in xs2])
        coeffs2 = np.ones(n2 + 1)
        coeffs2[1:-1:2] = 4
        coeffs2[2:-2:2] = 2
        result2 = (h2 / 3.0) * float(coeffs2 @ ys2)
        error_estimate = abs(result - result2) / 15.0
    else:
        error_estimate = h**4 * abs(b - a) / 180.0

    return float(result), float(error_estimate)


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature
# ---------------------------------------------------------------------------

# Precomputed nodes and weights for orders 2-5 on [-1, 1].
# These are exact to machine precision; no scipy dependency.
_GL_NODES_WEIGHTS: dict[int, tuple[list[float], list[float]]] = {
    2: (
        [-0.5773502691896257, 0.5773502691896257],
        [1.0, 1.0],
    ),
    3: (
        [-0.7745966692414834, 0.0, 0.7745966692414834],
        [0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
    ),
    4: (
        [
            -0.8611363115940526,
            -0.3399810435848563,
            0.3399810435848563,
            0.8611363115940526,
        ],
        [
            0.3478548451374538,
            0.6521451548625461,
            0.6521451548625461,
            0.3478548451374538,
        ],
    ),
    5: (
        [
            -0.9061798459386640,
            -0.5384693101056831,
            0.0,
            0.5384693101056831,
            0.9061798459386640,
        ],
        [
            0.2369268850561891,
            0.4786286704993665,
            0.5688888888888889,
            0.4786286704993665,
            0.2369268850561891,
        ],
    ),
}


def gauss_legendre(
    f: Callable[[float], float],
    a: float,
    b: float,
    order: int = 5,
) -> tuple[float, float]:
    """Approximate the integral of f from a to b using Gauss-Legendre quadrature.

    Exact for polynomials of degree <= 2*order - 1. Higher order means fewer
    function evaluations for smooth integrands compared to trapezoid/Simpson.

    For order > 5 the nodes and weights are computed numerically via the
    eigenvalue method on the Golub-Welsch tridiagonal matrix.

    Args:
        f: Integrand, callable on [a, b].
        a: Left endpoint.
        b: Right endpoint.
        order: Number of quadrature nodes (2 to 20).

    Returns:
        (result, error_estimate) where error_estimate is the absolute
        difference between this result and the order-1 result.

    Raises:
        ValueError: If order < 2 or a >= b.
    """
    if order < 2:
        raise ValueError(f"order must be >= 2; got {order}")
    if a >= b:
        raise ValueError(f"Require a < b; got a={a}, b={b}")

    nodes, weights = _gl_nodes_weights(order)

    # Change of variables: x = (b+a)/2 + (b-a)/2 * t, t in [-1, 1]
    scale = (b - a) / 2.0
    shift = (a + b) / 2.0
    result = scale * sum(w * f(shift + scale * t) for t, w in zip(nodes, weights))

    # Error estimate: compare to order-1 result
    if order > 2:
        nodes_lo, weights_lo = _gl_nodes_weights(order - 1)
        result_lo = scale * sum(
            w * f(shift + scale * t) for t, w in zip(nodes_lo, weights_lo)
        )
        error_estimate = abs(result - result_lo)
    else:
        error_estimate = float("inf")

    return float(result), float(error_estimate)


def _gl_nodes_weights(order: int) -> tuple[list[float], list[float]]:
    """Return Gauss-Legendre nodes and weights for the given order on [-1, 1].

    Uses precomputed tables for orders 2-5; falls back to the Golub-Welsch
    eigenvalue method for higher orders.
    """
    if order in _GL_NODES_WEIGHTS:
        return _GL_NODES_WEIGHTS[order]

    # Golub-Welsch: eigenvalues of the symmetric tridiagonal Jacobi matrix
    betas = np.array(
        [k / np.sqrt(4 * k**2 - 1) for k in range(1, order)], dtype=float
    )
    J = np.diag(betas, -1) + np.diag(betas, 1)
    nodes_arr, vecs = np.linalg.eigh(J)
    weights_arr = 2.0 * vecs[0, :] ** 2
    idx = np.argsort(nodes_arr)
    return nodes_arr[idx].tolist(), weights_arr[idx].tolist()
