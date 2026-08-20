"""Dynamic programming algorithms: LCS, 0/1 knapsack, edit distance, matrix chain, coin change."""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Longest Common Subsequence
# ---------------------------------------------------------------------------


def longest_common_subsequence(
    s1: str, s2: str
) -> tuple[int, list[list[int]]]:
    """Compute the length of the longest common subsequence of s1 and s2.

    Uses the standard O(mn) DP table. An LCS is not necessarily unique;
    this returns only the length and the table for backtracking.

    Args:
        s1: First sequence (string or any sequence type coerced to list).
        s2: Second sequence.

    Returns:
        (length, table) where table[i][j] = LCS length of s1[:i] and s2[:j].
    """
    m, n = len(s1), len(s2)
    table: list[list[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    return table[m][n], table


def lcs_backtrack(s1: str, s2: str, table: list[list[int]]) -> str:
    """Reconstruct one LCS string from the DP table produced by longest_common_subsequence.

    Args:
        s1: First sequence.
        s2: Second sequence.
        table: DP table from longest_common_subsequence.

    Returns:
        One longest common subsequence as a string.
    """
    result: list[str] = []
    i, j = len(s1), len(s2)
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            result.append(s1[i - 1])
            i -= 1
            j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(result))


# ---------------------------------------------------------------------------
# 0/1 Knapsack
# ---------------------------------------------------------------------------


def knapsack(
    weights: list[int],
    values: list[int],
    capacity: int,
) -> tuple[int, list[int]]:
    """Solve the 0/1 knapsack problem.

    Each item can be taken at most once. Uses a 2-D DP table with O(n * capacity)
    time and space.

    Args:
        weights: Item weights (positive integers).
        values: Item values (non-negative integers).
        capacity: Maximum weight the knapsack can hold.

    Returns:
        (max_value, selected_items) where max_value is the optimal total value
        and selected_items is a list of 0-indexed item indices in the optimal set.

    Raises:
        ValueError: If weights and values have different lengths.
    """
    n = len(weights)
    if len(values) != n:
        raise ValueError(
            f"weights and values must have the same length; got {n} and {len(values)}"
        )

    # dp[i][w] = max value using items 0..i-1 with capacity w
    dp: list[list[int]] = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w_i = weights[i - 1]
        v_i = values[i - 1]
        for w in range(capacity + 1):
            if w_i > w:
                dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - w_i] + v_i)

    # Backtrack to find selected items
    selected: list[int] = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= weights[i - 1]
    selected.reverse()

    return dp[n][capacity], selected


# ---------------------------------------------------------------------------
# Edit Distance (Levenshtein)
# ---------------------------------------------------------------------------


def edit_distance(s1: str, s2: str) -> tuple[int, list[list[int]]]:
    """Compute the Levenshtein edit distance between s1 and s2.

    Allows three operations, each costing 1: insert, delete, substitute.
    Uses the standard O(mn) DP table.

    Args:
        s1: Source string.
        s2: Target string.

    Returns:
        (distance, table) where distance is the minimum number of operations
        and table[i][j] = edit distance between s1[:i] and s2[:j].
    """
    m, n = len(s1), len(s2)
    table: list[list[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        table[i][0] = i
    for j in range(n + 1):
        table[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            table[i][j] = min(
                table[i - 1][j] + 1,       # delete from s1
                table[i][j - 1] + 1,       # insert into s1
                table[i - 1][j - 1] + cost,  # substitute
            )

    return table[m][n], table


# ---------------------------------------------------------------------------
# Matrix Chain Multiplication Order
# ---------------------------------------------------------------------------


def matrix_chain_order(dims: list[int]) -> tuple[int, list[list[int]]]:
    """Find the optimal parenthesization of a matrix chain product.

    Given n matrices A_1, ..., A_n where A_i has shape dims[i-1] x dims[i],
    finds the split points that minimize the total number of scalar multiplications.

    Args:
        dims: List of n+1 integers where matrix A_i has shape dims[i-1] x dims[i].

    Returns:
        (min_ops, split) where min_ops is the minimum number of scalar
        multiplications and split[i][j] is the optimal split point k for
        the subchain A_{i+1} ... A_{j+1} (0-indexed into dims).

    Raises:
        ValueError: If dims has fewer than 2 elements.
    """
    if len(dims) < 2:
        raise ValueError(f"dims must have at least 2 elements; got {len(dims)}")

    n = len(dims) - 1  # number of matrices
    INF = float("inf")

    # cost[i][j] = min scalar multiplications to compute product of A_{i+1}..A_{j+1}
    cost: list[list[float]] = [[0.0] * n for _ in range(n)]
    split: list[list[int]] = [[0] * n for _ in range(n)]

    for chain_len in range(2, n + 1):  # chain lengths 2..n
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            cost[i][j] = INF
            for k in range(i, j):
                q = cost[i][k] + cost[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if q < cost[i][j]:
                    cost[i][j] = q
                    split[i][j] = k

    return int(cost[0][n - 1]), split


# ---------------------------------------------------------------------------
# Coin Change (minimum coins)
# ---------------------------------------------------------------------------


def coin_change(coins: list[int], amount: int) -> tuple[int, list[int]]:
    """Find the minimum number of coins that sum to amount.

    Uses a 1-D DP table. Coin values can be used any number of times
    (unbounded knapsack variant).

    Args:
        coins: List of coin denominations (positive integers).
        amount: Target sum (non-negative integer).

    Returns:
        (min_coins, dp_table) where min_coins is the minimum number of coins
        needed (-1 if no solution exists) and dp_table[i] = minimum coins for
        sum i.

    Raises:
        ValueError: If amount < 0 or coins is empty.
    """
    if amount < 0:
        raise ValueError(f"amount must be >= 0; got {amount}")
    if not coins:
        raise ValueError("coins list must be non-empty")

    INF = amount + 1
    dp: list[int] = [INF] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for c in coins:
            if c <= i and dp[i - c] + 1 < dp[i]:
                dp[i] = dp[i - c] + 1

    min_coins = -1 if dp[amount] == INF else dp[amount]
    return min_coins, dp
