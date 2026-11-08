"""route_planner 单元测试：最短路矩阵、访问顺序优化、与暴力枚举对比。"""
import itertools
import math
import random
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from algo.route_planner import (  # noqa: E402
    build_matrix, exact_order, plan, route_length, shortest_distances,
    two_opt, nearest_neighbor_order,
)


def complete_graph(points):
    """完全图：边权 = 欧氏距离，A* 的启发函数因此是可采纳的。"""
    adj = {i: [] for i in range(len(points))}
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                continue
            d = math.dist(points[i], points[j])
            adj[i].append((j, d))
    return adj


def brute_force(matrix, start, targets):
    best = float("inf")
    best_order = None
    for perm in itertools.permutations(targets):
        length = route_length(matrix, start, list(perm))
        if length < best:
            best, best_order = length, list(perm)
    return best_order, best


def test_最短路矩阵与单源距离一致():
    adj = {0: [(1, 2.0)], 1: [(0, 2.0), (2, 3.0)], 2: [(1, 3.0)]}
    dist = shortest_distances(adj, 0)
    assert dist[2] == 5.0
    matrix = build_matrix(adj, [0, 1, 2])
    assert matrix[(0, 2)] == 5.0
    assert matrix[(2, 0)] == 5.0
    assert matrix[(1, 1)] == 0.0


def test_精确DP与暴力枚举结果一致():
    rng = random.Random(7)
    points = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(7)]
    adj = complete_graph(points)
    start, targets = 0, [1, 2, 3, 4, 5, 6]
    matrix = build_matrix(adj, [start] + targets)
    order, length = exact_order(matrix, start, targets)
    bf_order, bf_length = brute_force(matrix, start, targets)
    assert abs(length - bf_length) < 1e-6
    assert abs(route_length(matrix, start, order) - bf_length) < 1e-6


def test_2opt不劣于最近邻():
    rng = random.Random(11)
    points = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(9)]
    adj = complete_graph(points)
    start, targets = 0, list(range(1, 9))
    matrix = build_matrix(adj, [start] + targets)
    greedy = nearest_neighbor_order(matrix, start, targets)
    improved = two_opt(matrix, start, greedy)
    assert route_length(matrix, start, improved) <= route_length(matrix, start, greedy) + 1e-9


def test_规划覆盖全部目标且里程自洽():
    rng = random.Random(3)
    points = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(6)]
    adj = complete_graph(points)
    targets = [{"node": i, "kind": "fire", "label": f"火情 #{i}"} for i in range(1, 6)]
    result = plan(adj, points, 0, targets)
    assert result["ok"] is True
    assert len(result["stops"]) == 5
    assert {s["node"] for s in result["stops"]} == {1, 2, 3, 4, 5}
    legs_total = sum(leg["distance"] for leg in result["legs"])
    assert abs(legs_total - result["total_distance"]) < 0.5
    assert result["total_distance"] <= result["baseline_distance"] + 0.5
    assert result["path_nodes"][0] == 0
    assert result["algorithm"]["exact"] is True


def test_最近邻贪心可能被2opt改进():
    # 起点在左上，最近的点恰好在旁边，但「先近后远」会让后面的路绕回来：
    # 贪心 75.46 → 2-opt 62.16，说明局部搜索确实在优化访问顺序。
    points = [(0, 0), (28, 17), (2, 18), (13, 1), (0, 2), (6, 7), (16, 19)]
    adj = complete_graph(points)
    start, targets = 0, [1, 2, 3, 4, 5, 6]
    matrix = build_matrix(adj, [start] + targets)
    greedy = nearest_neighbor_order(matrix, start, targets)
    improved = two_opt(matrix, start, greedy)
    exact, exact_len = exact_order(matrix, start, targets)
    greedy_len = route_length(matrix, start, greedy)
    improved_len = route_length(matrix, start, improved)
    assert improved_len < greedy_len - 1e-6
    assert abs(route_length(matrix, start, exact) - exact_len) < 1e-6
    assert improved_len >= exact_len - 1e-9


def test_重复目标点只保留一个():
    adj = complete_graph([(0, 0), (10, 0), (20, 0)])
    targets = [{"node": 1, "label": "A"}, {"node": 1, "label": "B"}]
    result = plan(adj, [(0, 0), (10, 0), (20, 0)], 0, targets)
    assert len(result["stops"]) == 1


def test_不可达目标被剔除():
    adj = {0: [(1, 1.0)], 1: [(0, 1.0)], 2: [(3, 1.0)], 3: [(2, 1.0)]}
    xy = [(0, 0), (1, 0), (50, 50), (51, 50)]
    targets = [{"node": 1, "label": "可达"}, {"node": 2, "label": "孤岛"}]
    result = plan(adj, xy, 0, targets)
    assert result["ok"] is True
    assert [s["node"] for s in result["stops"]] == [1]
    assert result["dropped"] == ["孤岛"]


def test_没有目标时返回不可规划():
    adj = complete_graph([(0, 0), (10, 0)])
    result = plan(adj, [(0, 0), (10, 0)], 0, [])
    assert result["ok"] is False
