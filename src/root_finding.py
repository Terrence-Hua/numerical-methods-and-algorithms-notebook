"""Root-finding algorithms: bisection, Newton-Raphson, secant, Brent."""

from __future__ import annotations

from typing import Callable


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> tuple[float, int, list[float]]:
    """Find a root of f in [a, b] via bisection.

    Requires f(a) and f(b) to have opposite signs. Halves the interval
    each iteration, guaranteeing linear convergence.

    Args:
        f: Continuous function on [a, b].
        a: Left bracket.
        b: Right bracket.
        tol: Absolute tolerance on the interval half-width.
        max_iter: Maximum number of iterations.

    Returns:
        (root, iterations, errors) where errors[i] is the half-interval
        width after iteration i.

    Raises:
        ValueError: If f(a) and f(b) have the same sign.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(
            f"f(a) and f(b) must have opposite signs; got f({a})={fa:.4g}, f({b})={fb:.4g}"
        )

    errors: list[float] = []
    for i in range(max_iter):
        mid = (a + b) / 2.0
        fmid = f(mid)
        half_width = (b - a) / 2.0
        errors.append(half_width)

        if half_width < tol or fmid == 0.0:
            return mid, i + 1, errors

        if fa * fmid < 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid

    return (a + b) / 2.0, max_iter, errors


def newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[float, int, list[float]]:
    """Find a root of f using Newton-Raphson iteration.

    Converges quadratically near a simple root. Requires the derivative df.

    Args:
        f: Target function.
        df: Derivative of f.
        x0: Initial guess.
        tol: Absolute tolerance on |f(x)|.
        max_iter: Maximum number of iterations.

    Returns:
        (root, iterations, errors) where errors[i] = |f(x)| after iteration i.

    Raises:
        ZeroDivisionError: If df evaluates to zero during iteration.
    """
    x = x0
    errors: list[float] = []
    for i in range(max_iter):
        fx = f(x)
        errors.append(abs(fx))
        if abs(fx) < tol:
            return x, i, errors
        dfx = df(x)
        if dfx == 0.0:
            raise ZeroDivisionError(f"df(x) = 0 at x = {x}; Newton's method failed")
        x = x - fx / dfx

    return x, max_iter, errors


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[float, int, list[float]]:
    """Find a root of f using the secant method.

    Approximates the derivative via finite differences between the last two
    iterates. Converges superlinearly (order ~1.618) near a simple root.

    Args:
        f: Target function.
        x0: First initial guess.
        x1: Second initial guess.
        tol: Absolute tolerance on |f(x)|.
        max_iter: Maximum number of iterations.

    Returns:
        (root, iterations, errors) where errors[i] = |f(x)| after iteration i.
    """
    f0, f1 = f(x0), f(x1)
    errors: list[float] = [abs(f0), abs(f1)]

    for i in range(max_iter):
        if abs(f1) < tol:
            return x1, i + 2, errors
        denom = f1 - f0
        if denom == 0.0:
            # Stagnation — return best estimate
            return x1, i + 2, errors
        x2 = x1 - f1 * (x1 - x0) / denom
        x0, f0 = x1, f1
        x1, f1 = x2, f(x2)
        errors.append(abs(f1))

    return x1, max_iter, errors


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> tuple[float, int, list[float]]:
    """Find a root of f in [a, b] using Brent's method.

    Combines bisection, secant, and inverse quadratic interpolation.
    Guaranteed to converge; superlinear near a simple root.

    Args:
        f: Continuous function on [a, b].
        a: Left bracket.
        b: Right bracket.
        tol: Absolute tolerance on the root estimate.
        max_iter: Maximum number of iterations.

    Returns:
        (root, iterations, errors) where errors[i] = half-interval width
        at iteration i.

    Raises:
        ValueError: If f(a) and f(b) have the same sign.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(
            f"f(a) and f(b) must have opposite signs; got f({a})={fa:.4g}, f({b})={fb:.4g}"
        )

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    mflag = True
    s = b
    d = 0.0
    errors: list[float] = []

    for i in range(max_iter):
        errors.append(abs(b - a) / 2.0)

        if abs(b - a) < tol or fb == 0.0:
            return b, i + 1, errors

        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            # Secant step
            s = b - fb * (b - a) / (fb - fa)

        # Conditions to fall back to bisection
        cond1 = not (3 * a + b) / 4.0 < s < b and not b < s < (3 * a + b) / 4.0
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2.0
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) / 2.0
        cond4 = mflag and abs(b - c) < tol
        cond5 = (not mflag) and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2.0
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d, c, fc = c, b, fb

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    return b, max_iter, errors
