"""Graph algorithms: BFS, DFS, Dijkstra, Bellman-Ford, Floyd-Warshall."""

from __future__ import annotations

from collections import deque
from typing import Any

INF = float("inf")

# Graph type: adjacency list mapping node -> list of (neighbor, weight) pairs.
# Nodes can be any hashable type; weights are floats.
AdjList = dict[Any, list[tuple[Any, float]]]


# ---------------------------------------------------------------------------
# Breadth-First Search
# ---------------------------------------------------------------------------


def bfs(graph: AdjList, start: Any) -> tuple[list[Any], dict[Any, float]]:
    """Traverse a graph breadth-first from start.

    Args:
        graph: Adjacency list. Weights are ignored (BFS treats edges as unit-cost).
        start: Starting node. Must be a key in graph.

    Returns:
        (order, distances) where order is the BFS visit sequence and
        distances[v] is the number of edges from start to v.

    Raises:
        KeyError: If start is not in graph.
    """
    if start not in graph:
        raise KeyError(f"start node {start!r} not in graph")

    visited: set[Any] = {start}
    order: list[Any] = [start]
    distances: dict[Any, float] = {start: 0.0}
    queue: deque[Any] = deque([start])

    while queue:
        node = queue.popleft()
        for neighbor, _ in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                order.append(neighbor)
                distances[neighbor] = distances[node] + 1.0
                queue.append(neighbor)

    return order, distances


# ---------------------------------------------------------------------------
# Depth-First Search
# ---------------------------------------------------------------------------


def dfs(graph: AdjList, start: Any) -> tuple[list[Any], dict[Any, int]]:
    """Traverse a graph depth-first from start (iterative, avoids recursion limit).

    Args:
        graph: Adjacency list.
        start: Starting node. Must be a key in graph.

    Returns:
        (order, finish_times) where order is the DFS visit sequence and
        finish_times[v] is the step at which v was fully explored.

    Raises:
        KeyError: If start is not in graph.
    """
    if start not in graph:
        raise KeyError(f"start node {start!r} not in graph")

    visited: set[Any] = set()
    order: list[Any] = []
    finish_times: dict[Any, int] = {}
    timer = [0]

    stack: list[tuple[Any, bool]] = [(start, False)]

    while stack:
        node, returning = stack.pop()
        if returning:
            timer[0] += 1
            finish_times[node] = timer[0]
            continue
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        # Push sentinel to record finish time
        stack.append((node, True))
        for neighbor, _ in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append((neighbor, False))

    return order, finish_times


# ---------------------------------------------------------------------------
# Dijkstra's shortest-path algorithm
# ---------------------------------------------------------------------------


def dijkstra(
    graph: AdjList, start: Any
) -> tuple[dict[Any, float], dict[Any, Any | None]]:
    """Find shortest paths from start to all reachable nodes using Dijkstra's algorithm.

    Requires non-negative edge weights.

    Args:
        graph: Adjacency list with non-negative weights.
        start: Source node. Must be a key in graph.

    Returns:
        (distances, predecessors) where distances[v] is the shortest-path
        distance from start to v and predecessors[v] is v's predecessor on
        the shortest path (None for start).

    Raises:
        KeyError: If start is not in graph.
    """
    if start not in graph:
        raise KeyError(f"start node {start!r} not in graph")

    import heapq

    distances: dict[Any, float] = {start: 0.0}
    predecessors: dict[Any, Any | None] = {start: None}
    heap: list[tuple[float, Any]] = [(0.0, start)]

    # Initialize all known nodes to INF
    for node in graph:
        if node != start:
            distances[node] = INF
            predecessors[node] = None

    while heap:
        dist_u, u = heapq.heappop(heap)
        if dist_u > distances[u]:
            continue
        for v, weight in graph.get(u, []):
            alt = dist_u + weight
            if v not in distances:
                distances[v] = INF
                predecessors[v] = None
            if alt < distances[v]:
                distances[v] = alt
                predecessors[v] = u
                heapq.heappush(heap, (alt, v))

    return distances, predecessors


def shortest_path(predecessors: dict[Any, Any | None], start: Any, end: Any) -> list[Any]:
    """Reconstruct the shortest path from start to end using the predecessors map.

    Args:
        predecessors: Output from dijkstra or bellman_ford.
        start: Source node.
        end: Destination node.

    Returns:
        List of nodes from start to end (inclusive), or empty list if unreachable.
    """
    path: list[Any] = []
    node = end
    while node is not None:
        path.append(node)
        if node == start:
            break
        node = predecessors.get(node)
    if not path or path[-1] != start:
        return []
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Bellman-Ford shortest-path algorithm
# ---------------------------------------------------------------------------


def bellman_ford(
    graph: AdjList, start: Any
) -> tuple[dict[Any, float], dict[Any, Any | None], bool]:
    """Find shortest paths from start using Bellman-Ford.

    Handles negative edge weights; detects negative-weight cycles.

    Args:
        graph: Adjacency list. Negative weights are allowed.
        start: Source node. Must be a key in graph.

    Returns:
        (distances, predecessors, has_negative_cycle) where distances[v] is
        the shortest-path distance (or -inf if v is reachable via a negative
        cycle), predecessors[v] is v's predecessor, and has_negative_cycle
        is True if any negative cycle is reachable from start.

    Raises:
        KeyError: If start is not in graph.
    """
    if start not in graph:
        raise KeyError(f"start node {start!r} not in graph")

    # Collect all edges
    all_edges: list[tuple[Any, Any, float]] = []
    nodes: set[Any] = set(graph.keys())
    for u, neighbors in graph.items():
        for v, w in neighbors:
            all_edges.append((u, v, w))
            nodes.add(v)

    distances: dict[Any, float] = {n: INF for n in nodes}
    predecessors: dict[Any, Any | None] = {n: None for n in nodes}
    distances[start] = 0.0

    n = len(nodes)

    # Relax edges n-1 times
    for _ in range(n - 1):
        updated = False
        for u, v, w in all_edges:
            if distances[u] < INF and distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
                predecessors[v] = u
                updated = True
        if not updated:
            break

    # Check for negative cycles (n-th relaxation)
    has_negative_cycle = False
    for u, v, w in all_edges:
        if distances[u] < INF and distances[u] + w < distances[v]:
            has_negative_cycle = True
            break

    return distances, predecessors, has_negative_cycle


# ---------------------------------------------------------------------------
# Floyd-Warshall all-pairs shortest paths
# ---------------------------------------------------------------------------


def floyd_warshall(
    matrix: list[list[float]],
) -> tuple[list[list[float]], list[list[int]]]:
    """Compute all-pairs shortest paths using Floyd-Warshall.

    Args:
        matrix: n x n distance matrix. matrix[i][j] is the edge weight from
                node i to j, or float('inf') if there is no direct edge.
                Diagonal must be 0.

    Returns:
        (dist, next_node) where dist[i][j] is the shortest path distance from
        i to j, and next_node[i][j] is the next hop on the shortest path from i to j
        (-1 if no path exists).

    Raises:
        ValueError: If matrix is not square or diagonal is not all-zero.
    """
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be square")

    dist = [row[:] for row in matrix]
    next_node: list[list[int]] = [
        [j if matrix[i][j] < INF else -1 for j in range(n)] for i in range(n)
    ]
    for i in range(n):
        next_node[i][i] = i

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] < INF and dist[k][j] < INF:
                    through_k = dist[i][k] + dist[k][j]
                    if through_k < dist[i][j]:
                        dist[i][j] = through_k
                        next_node[i][j] = next_node[i][k]

    return dist, next_node


def fw_path(next_node: list[list[int]], i: int, j: int) -> list[int]:
    """Reconstruct the path from i to j using the next_node table from floyd_warshall.

    Returns:
        List of node indices from i to j, or empty list if no path exists.
    """
    if next_node[i][j] == -1:
        return []
    path = [i]
    while i != j:
        i = next_node[i][j]
        path.append(i)
    return path
