"""pathfinding.py —— 找路算法
- bfs / astar：传统网格版（保留给教学和旧测试）；
- graph_astar：任意加权图版，城市路网和人行道都用它。
"""
from collections import deque
from heapq import heappush, heappop
from math import hypot

STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # 上下左右四邻接


def bfs(grid, start, goal):
    """广度优先搜索。返回 [(row,col), ...] 从起点到终点；不可达返回 None"""
    h, w = len(grid), len(grid[0])
    queue = deque([start])
    parent = {start: None}                    # 记录「我从哪来」，顺便当「来过没」的登记本
    while queue:
        cur = queue.popleft()
        if cur == goal:                       # 找到了：沿着 parent 一路走回起点
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]                 # 反转成 起点→终点
        for dr, dc in STEPS:
            nxt = (cur[0] + dr, cur[1] + dc)
            if (0 <= nxt[0] < h and 0 <= nxt[1] < w
                    and grid[nxt[0]][nxt[1]] != 1
                    and nxt not in parent):
                parent[nxt] = cur
                queue.append(nxt)
    return None                               # 扩散完了都没碰到终点：不可达


def astar(grid, start, goal):
    """A*：BFS + 方向感。优先探索「离终点看起来更近」的格子"""
    h, w = len(grid), len(grid[0])

    def hx(p):                                # 启发函数：曼哈顿距离（可采纳，保证最优）
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    open_heap = [(hx(start), 0, start)]       # (f = 已走g+估计h, g, 位置)
    g = {start: 0}
    parent = {start: None}
    while open_heap:
        _, cur_g, cur = heappop(open_heap)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        for dr, dc in STEPS:
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < h and 0 <= nxt[1] < w and grid[nxt[0]][nxt[1]] != 1:
                ng = cur_g + 1
                if nxt not in g or ng < g[nxt]:
                    g[nxt] = ng
                    parent[nxt] = cur
                    heappush(open_heap, (ng + hx(nxt), ng, nxt))
    return None


def graph_astar(adj, start, goal, coords=None, blocked=()):
    """在加权图上跑 A*。

    adj:     {节点id: [(邻居id, 边长), ...]}
    coords:  {或列表}[节点id] -> (x, y)，用来计算欧氏启发距离
    blocked: 禁止经过的节点集合（可选）

    返回节点 id 列表（含起点和终点）；不可达返回 None。
    """
    if start == goal:
        return [start]
    if start in blocked or goal in blocked:
        return None

    def h(nid):
        if coords is None:
            return 0.0
        x, y = coords[nid]
        gx, gy = coords[goal]
        return hypot(x - gx, y - gy)

    open_heap = [(h(start), 0.0, start)]
    g = {start: 0.0}
    parent = {start: None}
    closed = set()
    while open_heap:
        _, cur_g, cur = heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        closed.add(cur)
        for nxt, cost in adj.get(cur, ()):
            if nxt in blocked or nxt in closed:
                continue
            ng = cur_g + cost
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                parent[nxt] = cur
                heappush(open_heap, (ng + h(nxt), ng, nxt))
    return None
