import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from algo.pathfinding import bfs, astar  # noqa: E402


def make_grid():
    grid = [[0] * 5 for _ in range(5)]
    for c in range(4):                         # 第 2 行留一堵墙（缺口在 col=4）
        grid[2][c] = 1
    return grid


def test_bfs_找到最短路():
    p = bfs(make_grid(), (0, 0), (4, 0))
    assert p is not None and len(p) == 13      # 必须绕墙，13 步


def test_astar_与bfs等长():
    p1 = bfs(make_grid(), (0, 0), (4, 0))
    p2 = astar(make_grid(), (0, 0), (4, 0))
    assert p1 and p2 and len(p1) == len(p2) == 13


def test_全堵死_不可达():
    grid = [[0] * 5 for _ in range(5)]
    for c in range(5):
        grid[2][c] = 1                         # 整行堵死
    assert bfs(grid, (0, 0), (4, 0)) is None
    assert astar(grid, (0, 0), (4, 0)) is None


def test_加权图A星_绕开高成本区域():
    from algo.pathfinding import graph_astar

    coords = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
    adj = {i: [] for i in range(9)}
    # 3x3 方格，中间 (1,1) 的入口成本很高
    for r in range(3):
        for c in range(3):
            nid = r * 3 + c
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if nr < 3 and nc < 3:
                    nxt = nr * 3 + nc
                    cost = 50 if (nid == 4 or nxt == 4) else 1
                    adj[nid].append((nxt, cost))
                    adj[nxt].append((nid, cost))
    path = graph_astar(adj, 0, 8, coords)
    assert path and path[0] == 0 and path[-1] == 8
    assert 4 not in path                        # 绕开中心高成本节点


def test_城市路网连通_所有站点可达():
    from sim.city import CityMap
    from algo.pathfinding import graph_astar

    city = CityMap()
    start = city.nearest_road_node(154, 300)
    for st in city.stations:
        assert graph_astar(city.road.adj, start, st["node"], city.road.xy) is not None
    for spot in city.fire_spots:
        assert graph_astar(city.road.adj, start, spot["node"], city.road.xy) is not None
