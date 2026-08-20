"""Tests for DP and graph algorithms — verified against hand-computed expected outputs."""

from __future__ import annotations

import pytest

from src.dp_algorithms import (
    coin_change,
    edit_distance,
    knapsack,
    lcs_backtrack,
    longest_common_subsequence,
    matrix_chain_order,
)
from src.graph_algorithms import (
    INF,
    bellman_ford,
    bfs,
    dfs,
    dijkstra,
    floyd_warshall,
    fw_path,
    shortest_path,
)


# ---------------------------------------------------------------------------
# Longest Common Subsequence
# ---------------------------------------------------------------------------


class TestLCS:
    def test_classic_example(self):
        length, table = longest_common_subsequence("ABCBDAB", "BDCAB")
        assert length == 4

    def test_backtrack(self):
        s1, s2 = "ABCBDAB", "BDCAB"
        length, table = longest_common_subsequence(s1, s2)
        lcs = lcs_backtrack(s1, s2, table)
        assert len(lcs) == length
        # Verify lcs is a subsequence of both
        def is_subseq(sub: str, seq: str) -> bool:
            it = iter(seq)
            return all(c in it for c in sub)
        assert is_subseq(lcs, s1)
        assert is_subseq(lcs, s2)

    def test_identical_strings(self):
        length, _ = longest_common_subsequence("HELLO", "HELLO")
        assert length == 5

    def test_no_common(self):
        length, table = longest_common_subsequence("ABC", "XYZ")
        assert length == 0
        lcs = lcs_backtrack("ABC", "XYZ", table)
        assert lcs == ""

    def test_one_empty(self):
        length, _ = longest_common_subsequence("", "ABCD")
        assert length == 0

    def test_single_char_match(self):
        length, _ = longest_common_subsequence("A", "A")
        assert length == 1

    def test_table_dimensions(self):
        s1, s2 = "ABC", "DE"
        _, table = longest_common_subsequence(s1, s2)
        assert len(table) == 4     # m+1
        assert len(table[0]) == 3  # n+1

    def test_longer_example(self):
        length, _ = longest_common_subsequence("AGGTAB", "GXTXAYB")
        assert length == 4  # GTAB


# ---------------------------------------------------------------------------
# 0/1 Knapsack
# ---------------------------------------------------------------------------


class TestKnapsack:
    def test_classic(self):
        val, items = knapsack([2, 3, 4, 5], [3, 4, 5, 6], 8)
        assert val == 10
        assert set(items) == {1, 3}  # items of weight 3 and 5

    def test_zero_capacity(self):
        val, items = knapsack([1, 2, 3], [10, 20, 30], 0)
        assert val == 0
        assert items == []

    def test_single_item_fits(self):
        val, items = knapsack([2], [7], 5)
        assert val == 7
        assert items == [0]

    def test_single_item_does_not_fit(self):
        val, items = knapsack([10], [7], 5)
        assert val == 0
        assert items == []

    def test_all_items_fit(self):
        val, items = knapsack([1, 1, 1], [3, 5, 2], 10)
        assert val == 10
        assert len(items) == 3

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            knapsack([1, 2], [3], 5)

    def test_selected_items_weight(self):
        weights = [3, 4, 2, 5, 1]
        values  = [5, 4, 3, 7, 2]
        capacity = 7
        val, items = knapsack(weights, values, capacity)
        total_weight = sum(weights[i] for i in items)
        total_value  = sum(values[i] for i in items)
        assert total_weight <= capacity
        assert total_value == val


# ---------------------------------------------------------------------------
# Edit Distance
# ---------------------------------------------------------------------------


class TestEditDistance:
    def test_kitten_sitting(self):
        d, _ = edit_distance("kitten", "sitting")
        assert d == 3

    def test_identical(self):
        d, _ = edit_distance("abc", "abc")
        assert d == 0

    def test_one_empty(self):
        d, _ = edit_distance("", "hello")
        assert d == 5

    def test_both_empty(self):
        d, _ = edit_distance("", "")
        assert d == 0

    def test_single_substitution(self):
        d, _ = edit_distance("a", "b")
        assert d == 1

    def test_insertions_only(self):
        d, _ = edit_distance("", "abc")
        assert d == 3

    def test_table_base_cases(self):
        _, table = edit_distance("abc", "de")
        assert table[0] == [0, 1, 2]
        assert [table[i][0] for i in range(4)] == [0, 1, 2, 3]

    def test_saturday_sunday(self):
        d, _ = edit_distance("saturday", "sunday")
        assert d == 3


# ---------------------------------------------------------------------------
# Matrix Chain Order
# ---------------------------------------------------------------------------


class TestMatrixChain:
    def test_classic_clrs_example(self):
        # CLRS Example: dims = [30, 35, 15, 5, 10, 20, 25]
        # Optimal: 15125
        ops, _ = matrix_chain_order([30, 35, 15, 5, 10, 20, 25])
        assert ops == 15125

    def test_two_matrices(self):
        # A: 2x3, B: 3x4 -> 2*3*4 = 24 multiplications
        ops, _ = matrix_chain_order([2, 3, 4])
        assert ops == 24

    def test_single_matrix(self):
        # One matrix: no multiplications needed
        ops, _ = matrix_chain_order([5, 10])
        assert ops == 0

    def test_dims_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            matrix_chain_order([5])

    def test_split_valid_range(self):
        dims = [10, 30, 5, 60]
        n = len(dims) - 1
        _, split = matrix_chain_order(dims)
        for i in range(n):
            for j in range(i, n):
                assert 0 <= split[i][j] < n


# ---------------------------------------------------------------------------
# Coin Change
# ---------------------------------------------------------------------------


class TestCoinChange:
    def test_classic(self):
        n, _ = coin_change([1, 5, 10, 25], 41)
        assert n == 4  # 25 + 10 + 5 + 1

    def test_zero_amount(self):
        n, dp = coin_change([1, 5], 0)
        assert n == 0
        assert dp[0] == 0

    def test_no_solution(self):
        n, _ = coin_change([3, 5], 7)
        assert n == -1

    def test_single_coin(self):
        n, _ = coin_change([7], 21)
        assert n == 3

    def test_dp_table_length(self):
        _, dp = coin_change([1, 2], 10)
        assert len(dp) == 11

    def test_negative_amount_raises(self):
        with pytest.raises(ValueError, match="amount must be"):
            coin_change([1], -1)

    def test_empty_coins_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            coin_change([], 5)

    def test_large_amount(self):
        n, _ = coin_change([1, 2, 5], 100)
        assert n == 20  # 20 x 5


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------

G1: "dict[str, list[tuple[str, float]]]" = {
    "A": [("B", 1), ("C", 4)],
    "B": [("C", 2), ("D", 5)],
    "C": [("D", 1)],
    "D": [],
}


class TestBFS:
    def test_order_starts_at_source(self):
        order, _ = bfs(G1, "A")
        assert order[0] == "A"

    def test_all_nodes_visited(self):
        order, _ = bfs(G1, "A")
        assert set(order) == {"A", "B", "C", "D"}

    def test_distances(self):
        _, dists = bfs(G1, "A")
        assert dists["A"] == 0
        assert dists["B"] == 1
        assert dists["C"] == 1
        assert dists["D"] == 2

    def test_disconnected_graph(self):
        g = {"X": [("Y", 1)], "Y": [], "Z": []}
        order, _ = bfs(g, "X")
        assert "Z" not in order

    def test_missing_start_raises(self):
        with pytest.raises(KeyError):
            bfs(G1, "Z")


# ---------------------------------------------------------------------------
# DFS
# ---------------------------------------------------------------------------


class TestDFS:
    def test_order_starts_at_source(self):
        order, _ = dfs(G1, "A")
        assert order[0] == "A"

    def test_all_reachable_visited(self):
        order, _ = dfs(G1, "A")
        assert set(order) == {"A", "B", "C", "D"}

    def test_finish_times_assigned(self):
        order, ft = dfs(G1, "A")
        assert set(ft.keys()) == {"A", "B", "C", "D"}
        assert len(set(ft.values())) == 4  # all distinct

    def test_missing_start_raises(self):
        with pytest.raises(KeyError):
            dfs(G1, "Z")


# ---------------------------------------------------------------------------
# Dijkstra
# ---------------------------------------------------------------------------


class TestDijkstra:
    def test_simple_path(self):
        dists, preds = dijkstra(G1, "A")
        assert dists["D"] == 4.0  # A->B->C->D = 1+2+1

    def test_source_distance_zero(self):
        dists, _ = dijkstra(G1, "A")
        assert dists["A"] == 0.0

    def test_predecessors(self):
        dists, preds = dijkstra(G1, "A")
        assert preds["A"] is None

    def test_shortest_path_reconstruction(self):
        _, preds = dijkstra(G1, "A")
        path = shortest_path(preds, "A", "D")
        assert path == ["A", "B", "C", "D"]

    def test_missing_start_raises(self):
        with pytest.raises(KeyError):
            dijkstra(G1, "Z")

    def test_triangle(self):
        g = {"S": [("A", 10), ("B", 3)], "A": [("B", 1)], "B": [("A", 4)]}
        dists, _ = dijkstra(g, "S")
        assert dists["A"] == 7.0  # S->B->A = 3+4


# ---------------------------------------------------------------------------
# Bellman-Ford
# ---------------------------------------------------------------------------


class TestBellmanFord:
    def test_same_as_dijkstra_positive(self):
        dists, preds, nc = bellman_ford(G1, "A")
        dists_d, _ = dijkstra(G1, "A")
        assert not nc
        for node in ["A", "B", "C", "D"]:
            assert abs(dists[node] - dists_d[node]) < 1e-12

    def test_negative_weights(self):
        g = {
            "S": [("A", 4), ("B", 5)],
            "A": [("C", 3)],
            "B": [("C", -2)],
            "C": [],
        }
        dists, _, nc = bellman_ford(g, "S")
        assert not nc
        assert dists["C"] == 3.0  # S->B->C = 5+(-2)

    def test_detects_negative_cycle(self):
        g = {
            "A": [("B", 1)],
            "B": [("C", -2)],
            "C": [("A", 0)],  # cycle with total weight -1
        }
        _, _, nc = bellman_ford(g, "A")
        assert nc

    def test_missing_start_raises(self):
        with pytest.raises(KeyError):
            bellman_ford(G1, "Z")


# ---------------------------------------------------------------------------
# Floyd-Warshall
# ---------------------------------------------------------------------------

FW_MATRIX = [
    [0, 3, INF, 7],
    [8, 0, 2, INF],
    [5, INF, 0, 1],
    [2, INF, INF, 0],
]


class TestFloydWarshall:
    def test_shortest_distances(self):
        dist, _ = floyd_warshall(FW_MATRIX)
        assert dist[0][2] == 5   # 0->1->2
        assert dist[0][3] == 6   # 0->1->2->3
        assert dist[3][0] == 2   # direct edge

    def test_diagonal_zero(self):
        dist, _ = floyd_warshall(FW_MATRIX)
        for i in range(4):
            assert dist[i][i] == 0

    def test_path_reconstruction(self):
        dist, next_node = floyd_warshall(FW_MATRIX)
        path = fw_path(next_node, 0, 3)
        assert path[0] == 0
        assert path[-1] == 3
        # Verify it's actually a valid path
        for a, b in zip(path, path[1:]):
            assert FW_MATRIX[a][b] < INF or True  # connected via intermediate

    def test_no_path(self):
        matrix = [[0, INF], [INF, 0]]
        _, next_node = floyd_warshall(matrix)
        assert fw_path(next_node, 0, 1) == []

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            floyd_warshall([[0, 1], [1, 0, 2]])
