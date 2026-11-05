"""city.py —— 城市路网与静态地图（仿真世界的“底图”）

职责：
- 定义道路、街区、建筑、树木、补水站、维修站、火情点位；
- 构建两张图：**车行图**（消防车走道路中心线）和**人行图**（行人走人行道 + 斑马线）；
- 提供 A* 寻路所需的邻接表与坐标。

坐标系：画布左上角为 (0, 0)，x 向右、y 向下，单位为“像素”。
所有位置都是连续坐标，不再有“第几格”的概念。
"""
from __future__ import annotations

import random
from collections import defaultdict


class Graph:
    """带坐标的加权无向图，节点 id 从 0 开始连续编号。"""

    def __init__(self):
        self.xy: list[tuple[float, float]] = []
        self.adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self.kinds: dict[tuple[int, int], str] = {}

    def add_node(self, x: float, y: float) -> int:
        nid = len(self.xy)
        self.xy.append((float(x), float(y)))
        return nid

    def add_edge(self, a: int, b: int, kind: str = "walk") -> None:
        if a == b:
            return
        (x1, y1), (x2, y2) = self.xy[a], self.xy[b]
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        self.adj[a].append((b, length))
        self.adj[b].append((a, length))
        self.kinds[(a, b)] = kind
        self.kinds[(b, a)] = kind

    def edge_kind(self, a: int, b: int) -> str | None:
        return self.kinds.get((a, b))


class CityMap:
    WIDTH = 1000
    HEIGHT = 740

    # 三条纵向主干道 + 三条横向主干道（中心线坐标）
    V_ROADS = (170, 500, 830)
    H_ROADS = (150, 380, 610)
    ROAD_W = 72          # 道路总宽度（双向两车道）
    SIDEWALK = 18        # 人行道宽度
    LANE_OFFSET = 16     # 车辆靠右行驶，距中心线的偏移
    WALK_OFFSET = 47     # 人行道中心线距道路中心线的距离（36 + 11）

    # 街区范围（道路之间的地块）
    BLOCKS = [
        (0, 134, 0, 114), (206, 464, 0, 114), (536, 794, 0, 114), (866, 1000, 0, 114),
        (0, 134, 186, 344), (206, 464, 186, 344), (536, 794, 186, 344), (866, 1000, 186, 344),
        (0, 134, 416, 574), (206, 464, 416, 574), (536, 794, 416, 574), (866, 1000, 416, 574),
        (0, 134, 646, 740), (206, 464, 646, 740), (536, 794, 646, 740), (866, 1000, 646, 740),
    ]

    # 火情点位：(门口 x, 门口 y, 道路停靠点 x, 道路停靠点 y)
    # 消防车停在道路上，隔着几十个单位向建筑门口喷水
    FIRE_SPOTS = [
        (128, 250, 170, 250), (212, 300, 170, 300),
        (462, 235, 500, 235), (540, 300, 500, 300),
        (792, 250, 830, 250), (872, 300, 830, 300),
        (300, 112, 300, 150), (650, 112, 650, 150),
        (300, 348, 300, 380), (700, 348, 700, 380),
        (300, 642, 300, 610), (700, 642, 700, 610),
    ]

    # 服务站点：kind = refill（补水）/ repair（维修），x,y 是图标位置，anchor 是道路停靠点
    STATIONS = [
        {"kind": "refill", "name": "西区补水站", "x": 224, "y": 265, "anchor": (170, 265)},
        {"kind": "refill", "name": "东区补水站", "x": 864, "y": 495, "anchor": (830, 495)},
        {"kind": "repair", "name": "中心维修站", "x": 545, "y": 128, "anchor": (500, 150)},
    ]

    def __init__(self):
        rng = random.Random(20261106)
        self.buildings: list[dict] = []
        self.trees: list[dict] = []
        self._build_buildings(rng)
        self._build_trees(rng)

        self.road = Graph()
        self.road_nodes: dict[tuple[float, float], int] = {}
        self.stations: list[dict] = []
        self.fire_spots: list[dict] = []
        self._build_road_graph()
        self._build_ped_graph()

    # ---------------------------------------------------------------- 静态装饰
    def _build_buildings(self, rng: random.Random) -> None:
        palette = ["#1e3a5f", "#243b53", "#2b4a6f", "#1c3550", "#2f4b63"]
        for (x0, x1, y0, y1) in self.BLOCKS:
            w, h = x1 - x0, y1 - y0
            if w > 180:
                gap = 12
                half = (w - gap - 20) / 2
                parts = [(x0 + 10, half), (x0 + 10 + half + gap, half)]
            else:
                parts = [(x0 + 10, w - 20)]
            for (bx, bw) in parts:
                bh = h - 20 - rng.randint(0, 12)
                if bw <= 8 or bh <= 8:
                    continue
                self.buildings.append({
                    "x": round(bx, 1), "y": round(y0 + 10, 1),
                    "w": round(bw, 1), "h": round(bh, 1),
                    "color": rng.choice(palette),
                    "floors": max(2, int(bh // 26)),
                    "cols": max(2, int(bw // 24)),
                })

    def _build_trees(self, rng: random.Random) -> None:
        for (x0, x1, y0, y1) in self.BLOCKS:
            for _ in range(2):
                side = rng.choice(("top", "bottom", "left", "right"))
                if side == "top":
                    x, y = rng.uniform(x0 + 12, x1 - 12), y0 - 6
                elif side == "bottom":
                    x, y = rng.uniform(x0 + 12, x1 - 12), y1 + 6
                elif side == "left":
                    x, y = x0 - 6, rng.uniform(y0 + 12, y1 - 12)
                else:
                    x, y = x1 + 6, rng.uniform(y0 + 12, y1 - 12)
                self.trees.append({"x": round(x, 1), "y": round(y, 1),
                                   "r": round(rng.uniform(7, 11), 1)})

    # ---------------------------------------------------------------- 车行图
    def _node(self, x: float, y: float) -> int:
        key = (round(float(x), 1), round(float(y), 1))
        if key not in self.road_nodes:
            self.road_nodes[key] = self.road.add_node(*key)
        return self.road_nodes[key]

    def _build_road_graph(self) -> None:
        # 1) 交叉口节点
        for x in self.V_ROADS:
            for y in self.H_ROADS:
                self._node(x, y)

        # 2) 服务站点、火情点位的道路停靠节点
        for st in self.STATIONS:
            ax, ay = st["anchor"]
            st["node"] = self._node(ax, ay)
            self.stations.append(dict(st))
        for spot in self.FIRE_SPOTS:
            door_x, door_y, anchor_x, anchor_y = spot
            self.fire_spots.append({
                "x": float(door_x), "y": float(door_y),
                "anchor": (float(anchor_x), float(anchor_y)),
                "node": self._node(anchor_x, anchor_y),
            })

        # 3) 沿每条道路把相邻节点连成边
        for x in self.V_ROADS:
            nodes = sorted((nid for (nx, ny), nid in self.road_nodes.items()
                            if abs(nx - x) < 0.5), key=lambda n: self.road.xy[n][1])
            for a, b in zip(nodes, nodes[1:]):
                self.road.add_edge(a, b, "road")
        for y in self.H_ROADS:
            nodes = sorted((nid for (nx, ny), nid in self.road_nodes.items()
                            if abs(ny - y) < 0.5), key=lambda n: self.road.xy[n][0])
            for a, b in zip(nodes, nodes[1:]):
                self.road.add_edge(a, b, "road")

    def nearest_road_node(self, x: float, y: float) -> int:
        """离 (x, y) 最近的道路节点，作为一次寻路的起点。"""
        best, best_d = 0, float("inf")
        for nid, (nx, ny) in enumerate(self.road.xy):
            d = (nx - x) ** 2 + (ny - y) ** 2
            if d < best_d:
                best, best_d = nid, d
        return best

    # ---------------------------------------------------------------- 人行图
    def _build_ped_graph(self) -> None:
        self.ped = Graph()
        self.ped_nodes: dict[tuple[float, float], int] = {}
        self.ped_edge_kind: dict[tuple[int, int], str] = {}

        def node(x: float, y: float) -> int:
            key = (round(float(x), 1), round(float(y), 1))
            if key not in self.ped_nodes:
                self.ped_nodes[key] = self.ped.add_node(*key)
            return self.ped_nodes[key]

        def edge(a: int, b: int, kind: str) -> None:
            self.ped.add_edge(a, b, kind)
            self.ped_edge_kind[(a, b)] = kind
            self.ped_edge_kind[(b, a)] = kind

        off = self.WALK_OFFSET
        # 每个交叉口的四个街角
        for X in self.V_ROADS:
            for Y in self.H_ROADS:
                for dx in (-off, off):
                    for dy in (-off, off):
                        node(X + dx, Y + dy)

        # 斑马线：横穿纵向道路（东西向行走） / 横穿横向道路（南北向行走）
        for X in self.V_ROADS:
            for Y in self.H_ROADS:
                edge(node(X - off, Y - off), node(X + off, Y - off), "cross_ew")
                edge(node(X - off, Y + off), node(X + off, Y + off), "cross_ew")
                edge(node(X - off, Y - off), node(X - off, Y + off), "cross_ns")
                edge(node(X + off, Y - off), node(X + off, Y + off), "cross_ns")

        # 人行道：相邻交叉口之间，沿街区四周
        for Y in self.H_ROADS:
            for i in range(len(self.V_ROADS) - 1):
                for dy in (-off, off):
                    edge(node(self.V_ROADS[i] + off, Y + dy),
                         node(self.V_ROADS[i + 1] - off, Y + dy), "walk")
        for X in self.V_ROADS:
            for j in range(len(self.H_ROADS) - 1):
                for dx in (-off, off):
                    edge(node(X + dx, self.H_ROADS[j] + off),
                         node(X + dx, self.H_ROADS[j + 1] - off), "walk")

    def random_ped_node(self, rng: random.Random) -> int:
        return rng.randrange(len(self.ped.xy))

    def ped_route(self, start: int, goal: int) -> list[int]:
        from algo.pathfinding import graph_astar
        return graph_astar(self.ped.adj, start, goal, self.ped.xy) or []

    # ---------------------------------------------------------------- 导出给前端
    def to_dict(self) -> dict:
        return {
            "width": self.WIDTH,
            "height": self.HEIGHT,
            "roads": {"v": list(self.V_ROADS), "h": list(self.H_ROADS)},
            "road_width": self.ROAD_W,
            "sidewalk": self.SIDEWALK,
            "walk_offset": self.WALK_OFFSET,
            "blocks": [{"x": a, "y": c, "w": b - a, "h": d - c}
                       for (a, b, c, d) in self.BLOCKS],
            "buildings": self.buildings,
            "trees": self.trees,
            "stations": [{"kind": s["kind"], "name": s["name"],
                          "x": s["x"], "y": s["y"]} for s in self.stations],
            "fire_spots": [{"x": f["x"], "y": f["y"]} for f in self.fire_spots],
            "intersections": [{"x": x, "y": y} for x in self.V_ROADS for y in self.H_ROADS],
        }
