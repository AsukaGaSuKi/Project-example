"""route_planner.py —— AI 最优路线规划

面向「一次出车、多个目的地」的路径优化问题，分三层：

1. **最短路层**：复用 `graph_astar`，在车行路网上算出任意两个目标点之间的最短路径；
2. **访问顺序层**：目标是「从当前位置出发，走遍所有待处理点，顺序不限」——
   这是一个开放式的旅行商问题（Open-TSP）。模块提供三种解法做对比：
   - 原始顺序：按火情出现的先后走（人的直觉，通常绕路）；
   - 最近邻 + 2-opt：先贪心再用「交换两条边」的方式局部改良；
   - 状态压缩 DP（Held-Karp）：目标点 ≤ 10 个时直接求精确最优解。
3. **解释层**：把「省了多少米、省了百分之几、访问顺序为什么这样排」整理成
   前端可以直接展示的结构，方便在答辩时讲清楚算法做了什么。

所有函数都是纯函数，不依赖 World，方便单元测试。
"""
from __future__ import annotations

from heapq import heappop, heappush

from algo.pathfinding import graph_astar

EXACT_LIMIT = 10          # 目标点不超过这个数量时用动态规划求精确最优
INF = float("inf")


# --------------------------------------------------------------------------- 最短路
def shortest_distances(adj, start: int) -> dict[int, float]:
    """Dijkstra：从 start 出发到全图每个节点的最短距离（路网很小，一次算完）。"""
    dist = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        d, cur = heappop(heap)
        if d > dist.get(cur, INF):
            continue
        for nxt, w in adj.get(cur, ()):
            nd = d + w
            if nd < dist.get(nxt, INF):
                dist[nxt] = nd
                heappush(heap, (nd, nxt))
    return dist


def build_matrix(adj, nodes: list[int]) -> dict[tuple[int, int], float]:
    """多源最短路矩阵，只保留 nodes 之间的元素。不可达记为 inf。"""
    wanted = set(nodes)
    matrix: dict[tuple[int, int], float] = {}
    for s in wanted:
        d = shortest_distances(adj, s)
        for g in wanted:
            matrix[(s, g)] = d.get(g, INF)
        matrix[(s, s)] = 0.0
    return matrix


# --------------------------------------------------------------------------- 顺序求解
def route_length(matrix: dict, start: int, order: list[int]) -> float:
    """start → order[0] → ... 的总长度。"""
    total, cur = 0.0, start
    for node in order:
        total += matrix.get((cur, node), INF)
        cur = node
    return total


def nearest_neighbor_order(matrix: dict, start: int, targets: list[int]) -> list[int]:
    """贪心最近邻：每一步都去「当前离我最近」的目标点。"""
    remaining = list(targets)
    order: list[int] = []
    cur = start
    while remaining:
        nxt = min(remaining, key=lambda n: (matrix.get((cur, n), INF), n))
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return order


def two_opt(matrix: dict, start: int, order: list[int]) -> list[int]:
    """2-opt 局部搜索：翻转一段访问顺序，能变短就接受，直到没有改进。

    开放式路径（不需要回到起点）的一小段顺序反转，等价于把两条边拆开重连。
    """
    best = list(order)
    best_len = route_length(matrix, start, best)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best) + 1):
                candidate = best[:i] + best[i:j][::-1] + best[j:]
                length = route_length(matrix, start, candidate)
                if length + 1e-9 < best_len:
                    best, best_len = candidate, length
                    improved = True
    return best


def exact_order(matrix: dict, start: int, targets: list[int]) -> tuple[list[int], float]:
    """Held-Karp 状态压缩 DP，求开放路径的精确最优访问顺序。

    dp[(mask, last)] = 从起点出发、已经访问 mask 这些点、最后停在 last 的最短距离。
    目标点 10 个时状态数约 10 * 2^10，毫秒级完成。
    """
    n = len(targets)
    dp: dict[tuple[int, int], float] = {}
    parent: dict[tuple[int, int], int] = {}
    for i, node in enumerate(targets):
        dp[(1 << i, i)] = matrix[(start, node)]
        parent[(1 << i, i)] = -1

    for mask in range(1 << n):
        for last in range(n):
            key = (mask, last)
            if key not in dp:
                continue
            base = dp[key]
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                nm = mask | (1 << nxt)
                cand = base + matrix[(targets[last], targets[nxt])]
                if cand < dp.get((nm, nxt), INF) - 1e-9:
                    dp[(nm, nxt)] = cand
                    parent[(nm, nxt)] = last

    full = (1 << n) - 1
    best_last = min(range(n), key=lambda i: dp.get((full, i), INF))
    best_len = dp[(full, best_last)]

    order = []
    mask, last = full, best_last
    while last != -1:
        order.append(targets[last])
        prev = parent[(mask, last)]
        mask ^= 1 << last
        last = prev
    order.reverse()
    return order, best_len


# --------------------------------------------------------------------------- 对外入口
def plan(
    adj,
    xy: list[tuple[float, float]],
    start: int,
    targets: list[dict],
    *,
    start_point: tuple[float, float] | None = None,
) -> dict:
    """规划一次「多点访问」的最优路线。

    targets: [{"node": int, "kind": "fire"|"refill"|"repair"|"patrol", ...}, ...]
             其余字段原样透传，便于前端显示；相同 node 只保留第一个。
    start_point: 车辆真实坐标，仅用于给前端画「车辆 → 第一个路口」的引导线。

    返回：访问顺序、每一段的距离、总距离、三种算法对比、完整节点路径。
    """
    # 同一路口只去一次（多个目标可能共用停靠点）
    seen: set[int] = set()
    uniq: list[dict] = []
    for t in targets:
        if t["node"] in seen:
            continue
        seen.add(t["node"])
        uniq.append(t)
    if not uniq:
        return {"ok": False, "reason": "当前没有需要规划的目标点"}

    nodes = [start] + [t["node"] for t in uniq]
    matrix = build_matrix(adj, nodes)

    # 不可达的目标直接剔除，避免整条路线作废
    reachable = [t for t in uniq if matrix[(start, t["node"])] < INF]
    reachable_nodes = {t["node"] for t in reachable}
    dropped = [t for t in uniq if t["node"] not in reachable_nodes]
    if not reachable:
        return {"ok": False, "reason": "所有目标点都不可达（可能被路障封锁）"}

    target_nodes = [t["node"] for t in reachable]
    baseline = target_nodes[:]                      # 原始顺序：按目标出现先后
    greedy = nearest_neighbor_order(matrix, start, target_nodes)
    improved = two_opt(matrix, start, greedy)

    exact_used = len(target_nodes) <= EXACT_LIMIT
    if exact_used:
        exact, exact_len = exact_order(matrix, start, target_nodes)
    else:
        exact, exact_len = None, INF

    candidates = [
        ("原始顺序", baseline, route_length(matrix, start, baseline)),
        ("最近邻贪心", greedy, route_length(matrix, start, greedy)),
        ("最近邻 + 2-opt", improved, route_length(matrix, start, improved)),
    ]
    if exact_used:
        candidates.append(("状态压缩 DP（精确最优）", exact, exact_len))
    best_len = min(c[2] for c in candidates)
    # 同样长时优先用更强的算法署名（精确 > 2-opt > 贪心 > 原始）
    label, order_nodes, total = [c for c in candidates if abs(c[2] - best_len) < 1e-9][-1]

    by_node = {t["node"]: t for t in reachable}
    stops = []
    legs = []
    path_nodes: list[int] = []
    cur = start
    cumulative = 0.0
    for i, node in enumerate(order_nodes):
        leg_dist = matrix[(cur, node)]
        leg_path = graph_astar(adj, cur, node, xy)
        if leg_path is None:                        # 理论上不会发生：矩阵里可达
            leg_path = [cur, node]
        # 相邻两段共享路口，拼接时去重
        path_nodes.extend(leg_path if not path_nodes else leg_path[1:])
        cumulative += leg_dist
        target = by_node[node]
        stop = {
            "order": i + 1,
            "node": node,
            "x": round(xy[node][0], 1),
            "y": round(xy[node][1], 1),
            "kind": target.get("kind", "target"),
            "label": target.get("label", f"目标 {i + 1}"),
            "fire_id": target.get("fire_id"),
            "intensity": target.get("intensity"),
            "leg_distance": round(leg_dist, 1),
            "distance_so_far": round(cumulative, 1),
        }
        stops.append(stop)
        legs.append({"from": cur, "to": node, "distance": round(leg_dist, 1),
                     "eta": round(leg_dist / 95.0, 1)})
        cur = node

    baseline_len = route_length(matrix, start, baseline)
    saved = baseline_len - total
    saved_pct = (saved / baseline_len * 100.0) if baseline_len > 0 else 0.0

    return {
        "ok": True,
        "mode": "multi",
        "start_node": start,
        "start_point": list(start_point) if start_point else list(xy[start]),
        "stops": stops,
        "legs": legs,
        "path_nodes": path_nodes,
        "total_distance": round(total, 1),
        "baseline_distance": round(baseline_len, 1),
        "saved_distance": round(saved, 1),
        "saved_percent": round(saved_pct, 1),
        "algorithm": {
            "name": label,
            "shortest_path": "A*（欧氏距离启发）",
            "optimizer": "状态压缩 DP" if exact_used else "最近邻 + 2-opt",
            "exact": exact_used,
            "targets": len(reachable),
            "candidates": [
                {"name": name, "distance": round(length, 1), "order": order}
                for name, order, length in candidates
            ],
        },
        "dropped": [t.get("label") for t in dropped],
        "eta": eta_for(total, len(stops)),
    }


def eta_for(distance: float, stops: int = 0, speed: float = 82.0) -> float:
    """粗略用时：按有效均速估算行驶时间，再加每个点的停靠、转向与作业时间。"""
    return distance / speed + 2.5 * stops


__all__ = [
    "plan", "build_matrix", "shortest_distances", "nearest_neighbor_order",
    "two_opt", "exact_order", "route_length", "eta_for",
]
