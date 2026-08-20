"""Tests for DP and graph algorithms."""

from __future__ import annotations

import math
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
    bellman_ford,
    bfs,
    dfs,
    dijkstra,
    floyd_warshall,
    fw_path,
    shortest_path,
)

INF = float("inf")


# ---------------------------------------------------------------------------
# LCS
# ---------------------------------------------------------------------------


class TestLCS:
    def test_known_pair(self):
        length, table = longest_common_subsequence("ABCBDAB", "BDCABA")
        assert length == 4

    def test_identical_strings(self):
        length, _ = longest_common_subsequence("hello", "hello")
        assert length == 5

    def test_no_common(self):
        length, _ = longest_common_subsequence("abc", "xyz")
        assert length == 0

    def test_empty_string(self):
        length, _ = longest_common_subsequence("", "abc")
        assert length == 0

    def test_single_char_match(self):
        length, _ = longest_common_subsequence("a", "a")
        assert length == 1

    def test_single_char_no_match(self):
        length, _ = longest_common_subsequence("a", "b")
        assert length == 0

    def test_backtrack_known(self):
        s1, s2 = "ABCBDAB", "BDCABA"
        length, table = longest_common_subsequence(s1, s2)
        lcs = lcs_backtrack(s1, s2, table)
        assert len(lcs) == length
        # Verify it is actually a subsequence of both
        def is_subseq(sub, seq):
            it = iter(seq)
            return all(c in it for c in sub)
        assert is_subseq(lcs, s1)
        assert is_subseq(lcs, s2)

    def test_backtrack_identical(self):
        s1 = "hello"
        length, table = longest_common_subsequence(s1, s1)
        lcs = lcs_backtrack(s1, s1, table)
        assert lcs == s1

    def test_table_dimensions(self):
        s1, s2 = "abc", "de"
        _, table = longest_common_subsequence(s1, s2)
        assert len(table) == len(s1) + 1
        assert len(table[0]) == len(s2) + 1


# ---------------------------------------------------------------------------
# Knapsack
# ---------------------------------------------------------------------------


class TestKnapsack:
    def test_basic(self):
        weights = [2, 3, 4, 5]
        values = [3, 4, 5, 6]
        max_val, items = knapsack(weights, values, 5)
        assert max_val == 7  # items 0 (w=2,v=3) + 1 (w=3,v=4)
        assert sum(weights[i] for i in items) <= 5
        assert sum(values[i] for i in items) == max_val

    def test_zero_capacity(self):
        max_val, items = knapsack([1, 2], [3, 4], 0)
        assert max_val == 0
        assert items == []

    def test_all_items_fit(self):
        weights = [1, 1, 1]
        values = [2, 3, 4]
        max_val, items = knapsack(weights, values, 10)
        assert max_val == 9
        assert sorted(items) == [0, 1, 2]

    def test_single_item_fits(self):
        max_val, items = knapsack([3], [5], 3)
        assert max_val == 5
        assert items == [0]

    def test_single_item_too_heavy(self):
        max_val, items = knapsack([5], [10], 3)
        assert max_val == 0
        assert items == []

    def test_selected_items_valid(self):
        weights = [1, 3, 4, 5]
        values = [1, 4, 5, 7]
        max_val, items = knapsack(weights, values, 7)
        assert sum(weights[i] for i in items) <= 7
        assert sum(values[i] for i in items) == max_val

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            knapsack([1, 2], [1], 5)


# ---------------------------------------------------------------------------
# Edit Distance
# ---------------------------------------------------------------------------


class TestEditDistance:
    def test_identical(self):
        dist, _ = edit_distance("abc", "abc")
        assert dist == 0

    def test_empty_to_string(self):
        dist, _ = edit_distance("", "abc")
        assert dist == 3

    def test_string_to_empty(self):
        dist, _ = edit_distance("abc", "")
        assert dist == 3

    def test_both_empty(self):
        dist, _ = edit_distance("", "")
        assert dist == 0

    def test_single_substitution(self):
        dist, _ = edit_distance("a", "b")
        assert dist == 1

    def test_known_pair(self):
        dist, _ = edit_distance("kitten", "sitting")
        assert dist == 3

    def test_known_pair_2(self):
        dist, _ = edit_distance("saturday", "sunday")
        assert dist == 3

    def test_table_base_cases(self):
        _, table = edit_distance("ab", "cd")
        assert table[0][0] == 0
        assert table[1][0] == 1
        assert table[2][0] == 2
        assert table[0][1] == 1
        assert table[0][2] == 2

    def test_one_insert(self):
        dist, _ = edit_distance("abc", "abcd")
        assert dist == 1

    def test_one_delete(self):
        dist, _ = edit_distance("abcd", "abc")
        assert dist == 1


# ---------------------------------------------------------------------------
# Matrix Chain Order
# ---------------------------------------------------------------------------


class TestMatrixChainOrder:
    def test_two_matrices(self):
        # A: 10x30, B: 30x5 -> 10*30*5 = 1500 ops
        ops, _ = matrix_chain_order([10, 30, 5])
        assert ops == 1500

    def test_three_matrices(self):
        # Classic example: 10x30, 30x5, 5x60
        # Option 1: (AB)C = 1500 + 10*5*60 = 1500 + 3000 = 4500
        # Option 2: A(BC) = 30*5*60 + 10*30*60 = 9000 + 18000 = 27000
        ops, _ = matrix_chain_order([10, 30, 5, 60])
        assert ops == 4500

    def test_four_matrices_classic(self):
        # CLRS example: dims [30,35,15,5,10,20,25] (6 matrices)
        # Known answer: 15125
        ops, _ = matrix_chain_order([30, 35, 15, 5, 10, 20, 25])
        assert ops == 15125

    def test_single_matrix(self):
        ops, _ = matrix_chain_order([5, 10])
        assert ops == 0

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            matrix_chain_order([5])

    def test_split_is_valid(self):
        dims = [10, 30, 5, 60]
        n = len(dims) - 1
        _, split = matrix_chain_order(dims)
        assert 0 <= split[0][n - 1] < n - 1


# ---------------------------------------------------------------------------
# Coin Change
# ---------------------------------------------------------------------------


class TestCoinChange:
    def test_basic(self):
        min_coins, _ = coin_change([1, 5, 10, 25], 36)
        assert min_coins == 3  # 25 + 10 + 1

    def test_exact_denomination(self):
        min_coins, _ = coin_change([1, 5, 10], 10)
        assert min_coins == 1

    def test_no_solution(self):
        min_coins, _ = coin_change([2], 3)
        assert min_coins == -1

    def test_zero_amount(self):
        min_coins, dp = coin_change([1, 5], 0)
        assert min_coins == 0
        assert dp[0] == 0

    def test_single_coin(self):
        min_coins, _ = coin_change([3], 9)
        assert min_coins == 3

    def test_negative_amount_raises(self):
        with pytest.raises(ValueError):
            coin_change([1], -1)

    def test_empty_coins_raises(self):
        with pytest.raises(ValueError):
            coin_change([], 5)

    def test_dp_table_length(self):
        amount = 10
        _, dp = coin_change([1, 5], amount)
        assert len(dp) == amount + 1

    def test_large_amount(self):
        min_coins, _ = coin_change([1, 5, 10, 25], 100)
        assert min_coins == 4  # four 25s


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------


def _simple_graph():
    return {
        0: [(1, 1.0), (2, 1.0)],
        1: [(3, 1.0)],
        2: [(3, 1.0), (4, 1.0)],
        3: [(5, 1.0)],
        4: [(5, 1.0)],
        5: [],
    }


class TestBFS:
    def test_visit_order_starts_with_start(self):
        g = _simple_graph()
        order, _ = bfs(g, 0)
        assert order[0] == 0

    def test_all_nodes_visited(self):
        g = _simple_graph()
        order, _ = bfs(g, 0)
        assert set(order) == set(g.keys())

    def test_distances(self):
        g = _simple_graph()
        _, dist = bfs(g, 0)
        assert dist[0] == 0
        assert dist[1] == 1
        assert dist[2] == 1
        assert dist[3] == 2
        assert dist[4] == 2
        assert dist[5] == 3

    def test_disconnected_graph(self):
        g = {0: [(1, 1.0)], 1: [], 2: [(3, 1.0)], 3: []}
        order, dist = bfs(g, 0)
        assert set(order) == {0, 1}
        assert 2 not in dist

    def test_invalid_start_raises(self):
        g = _simple_graph()
        with pytest.raises(KeyError):
            bfs(g, 99)

    def test_single_node(self):
        g = {0: []}
        order, dist = bfs(g, 0)
        assert order == [0]
        assert dist == {0: 0.0}


# ---------------------------------------------------------------------------
# DFS
# ---------------------------------------------------------------------------


class TestDFS:
    def test_start_node_first(self):
        g = _simple_graph()
        order, _ = dfs(g, 0)
        assert order[0] == 0

    def test_all_nodes_visited(self):
        g = _simple_graph()
        order, _ = dfs(g, 0)
        assert set(order) == set(g.keys())

    def test_finish_times_unique(self):
        g = _simple_graph()
        _, finish = dfs(g, 0)
        times = list(finish.values())
        assert len(times) == len(set(times))

    def test_invalid_start_raises(self):
        g = _simple_graph()
        with pytest.raises(KeyError):
            dfs(g, 99)

    def test_single_node(self):
        g = {0: []}
        order, finish = dfs(g, 0)
        assert order == [0]
        assert 0 in finish


# ---------------------------------------------------------------------------
# Dijkstra
# ---------------------------------------------------------------------------


def _weighted_graph():
    return {
        "A": [("B", 1.0), ("C", 4.0)],
        "B": [("C", 2.0), ("D", 5.0)],
        "C": [("D", 1.0)],
        "D": [],
    }


class TestDijkstra:
    def test_shortest_distances(self):
        g = _weighted_graph()
        dist, _ = dijkstra(g, "A")
        assert dist["A"] == 0.0
        assert dist["B"] == 1.0
        assert dist["C"] == 3.0
        assert dist["D"] == 4.0

    def test_predecessors(self):
        g = _weighted_graph()
        dist, pred = dijkstra(g, "A")
        assert pred["A"] is None
        assert pred["B"] == "A"
        assert pred["C"] == "B"
        assert pred["D"] == "C"

    def test_shortest_path_reconstruction(self):
        g = _weighted_graph()
        _, pred = dijkstra(g, "A")
        path = shortest_path(pred, "A", "D")
        assert path == ["A", "B", "C", "D"]

    def test_unreachable_node(self):
        g = {"A": [], "B": []}
        dist, pred = dijkstra(g, "A")
        assert dist["B"] == INF

    def test_invalid_start_raises(self):
        with pytest.raises(KeyError):
            dijkstra(_weighted_graph(), "Z")

    def test_single_node(self):
        g = {"A": []}
        dist, pred = dijkstra(g, "A")
        assert dist["A"] == 0.0
        assert pred["A"] is None

    def test_path_unreachable(self):
        g = {"A": [], "B": []}
        _, pred = dijkstra(g, "A")
        path = shortest_path(pred, "A", "B")
        assert path == []


# ---------------------------------------------------------------------------
# Bellman-Ford
# ---------------------------------------------------------------------------


class TestBellmanFord:
    def test_no_negative_cycle(self):
        g = {
            "s": [("u", 6.0), ("y", 7.0)],
            "u": [("x", 5.0), ("v", -4.0)],
            "v": [("s", 2.0)],
            "x": [("v", -2.0)],
            "y": [("u", 8.0), ("x", -3.0), ("v", 9.0)],
        }
        dist, _, neg_cycle = bellman_ford(g, "s")
        assert not neg_cycle
        # From CLRS Figure 24.4 (adjusted for matching edge set)
        assert dist["s"] == 0.0

    def test_detects_negative_cycle(self):
        g = {
            0: [(1, 1.0)],
            1: [(2, -3.0)],
            2: [(0, 1.0)],
        }
        _, _, neg_cycle = bellman_ford(g, 0)
        assert neg_cycle

    def test_simple_path(self):
        g = {"A": [("B", 2.0)], "B": [("C", 3.0)], "C": []}
        dist, _, neg_cycle = bellman_ford(g, "A")
        assert not neg_cycle
        assert dist["A"] == 0.0
        assert dist["B"] == 2.0
        assert dist["C"] == 5.0

    def test_invalid_start_raises(self):
        with pytest.raises(KeyError):
            bellman_ford(_weighted_graph(), "Z")

    def test_matches_dijkstra_on_positive_graph(self):
        g = _weighted_graph()
        bf_dist, _, _ = bellman_ford(g, "A")
        dj_dist, _ = dijkstra(g, "A")
        for node in dj_dist:
            assert abs(bf_dist[node] - dj_dist[node]) < 1e-12


# ---------------------------------------------------------------------------
# Floyd-Warshall
# ---------------------------------------------------------------------------


class TestFloydWarshall:
    def _sample_matrix(self):
        I = INF
        return [
            [0,   3,   I,   7],
            [8,   0,   2,   I],
            [5,   I,   0,   1],
            [2,   I,   I,   0],
        ]

    def test_known_answer(self):
        dist, _ = floyd_warshall(self._sample_matrix())
        assert dist[0][2] == 5   # 0->1->2
        assert dist[3][0] == 2   # direct edge

    def test_self_distances_zero(self):
        dist, _ = floyd_warshall(self._sample_matrix())
        for i in range(4):
            assert dist[i][i] == 0.0

    def test_symmetry_on_undirected(self):
        I = INF
        m = [
            [0, 1, I],
            [1, 0, 1],
            [I, 1, 0],
        ]
        dist, _ = floyd_warshall(m)
        for i in range(3):
            for j in range(3):
                assert dist[i][j] == dist[j][i]

    def test_path_reconstruction(self):
        dist, nxt = floyd_warshall(self._sample_matrix())
        path = fw_path(nxt, 0, 2)
        assert path[0] == 0
        assert path[-1] == 2
        # Check path length matches distance
        path_len = sum(
            self._sample_matrix()[path[k]][path[k + 1]]
            for k in range(len(path) - 1)
        )
        assert abs(path_len - dist[0][2]) < 1e-12

    def test_no_path(self):
        I = INF
        m = [[0, I], [I, 0]]
        _, nxt = floyd_warshall(m)
        path = fw_path(nxt, 0, 1)
        assert path == []

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            floyd_warshall([[0, 1], [1, 0, 2]])

    def test_2x2(self):
        m = [[0, 5.0], [3.0, 0]]
        dist, _ = floyd_warshall(m)
        assert dist[0][1] == 5.0
        assert dist[1][0] == 3.0
