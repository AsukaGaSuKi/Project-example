"""world.py —— 自动消防小车仿真内核

一台带机械臂的消防小车，在连续坐标的城市路网中完成：
- 火情检测 → 立刻自检（结果供前端可视化）→ 出发灭火；
- 水量低于 10% 自动前往最近补水站；灭火时允许用到 0% 再补水；
- 功能故障自动前往最近维修站；
- 行驶中遇到路障，停车用机械臂清障；
- 遇到行人自动降速慢行，行人通过后恢复；
- 行人随机生成，走人行道、过斑马线并遵守红绿灯；
- 路网常驻四辆私家车：路口让行、靠边避让、同侧排队等消防车起步，任何时候都不重叠。

所有状态由后端统一维护，update(dt) 是唯一推进仿真时间的入口，便于测试。
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

from algo.pathfinding import graph_astar
from algo.route_planner import plan as plan_multi_route
from sim.city import CityMap

BASE_SPEED = 95.0        # 正常巡航速度（单位/秒）
SLOW_SPEED = 26.0        # 遇到行人时的慢行速度
ACCEL = 120.0            # 加速度
TURN_RATE = 2.8          # 最大转向角速度（弧度/秒）
LANE = CityMap.LANE_OFFSET

WATER_LOW = 10.0         # 水量低阈值
REFILL_RATE = 18.0       # 补水速度（%/秒）
SPRAY_WATER = 7.0        # 灭火耗水（%/秒）
SPRAY_POWER = 24.0       # 灭火强度下降（点/秒）
REPAIR_TIME = 4.0
SELF_CHECK_STEP = 0.38   # 每个自检项目耗时
AI_SCAN_TIME = 1.4       # AI 巡检点在点停留时间（秒）
AI_ARRIVE = 42.0         # 判定「到达停靠点」的距离

# ---- 私家车（NPC）：常驻路网，四辆，给消防车让行 ----
CAR_MAX = 4              # 路网常驻私家车数量（不再随机刷新/消失）
CAR_SPEED_MIN = 52.0     # 私家车巡航速度区间（比消防车慢，便于被追上时让行）
CAR_SPEED_MAX = 68.0
CAR_ACCEL = 90.0
CAR_BRAKE = 160.0
CAR_TURN = 3.2
CAR_SPAWN_CLEAR = 90.0   # 出生点离消防车的最小距离
CAR_ROUTE_MIN = 320.0    # 优先给私家车派一条够长的路线，避免刚上路就到终点
CAR_YIELD_DIST = 150.0   # 消防车进入这个范围，私家车开始让行
CAR_CLEAR_DIST = 190.0   # 消防车退出这个范围，让行解除（带回滞，避免抖动）
CAR_INTERSECTION_DIST = 78.0   # 离路口多近算「在路口」
CAR_PULL_OVER = 30.0     # 让行时向路肩平移的距离（消防车很宽，要让到路肩才不重叠）
CAR_WAIT_GAP = 52.0      # 同侧等消防车时，停在它后面的距离
CAR_MIN_GAP = 34.0       # 小车与消防车中心的最小间距（硬约束，杜绝重叠）
CAR_CAR_GAP = 30.0       # 两辆小车中心的最小间距（硬约束，杜绝重叠）
CAR_SIDE_RATE = 26.0     # 横向平移速度

# 车身颜色：与消防车同款的深红，只做细微深浅区分
CAR_COLORS = ("#c8102e", "#a80f27", "#d7263d", "#9e0b1f")

FAULT_NAMES = {
    "arm": "机械臂",
    "pump": "灭火水泵",
    "sensor": "传感器",
    "nav": "导航定位",
}

SELF_CHECK_ITEMS = [
    ("water", "水量与水箱"),
    ("battery", "电池电量"),
    ("arm", "机械臂"),
    ("pump", "灭火水泵"),
    ("sensor", "传感器阵列"),
    ("nav", "导航与定位"),
]


@dataclass
class Truck:
    x: float
    y: float
    heading: float
    speed: float = 0.0
    water: float = 100.0
    battery: float = 100.0
    state: str = "idle"
    route: list[int] = field(default_factory=list)
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    wp_index: int = 0
    target_fire: int | None = None
    target_station: int | None = None
    slowing: bool = False
    fault: str | None = None
    arm_target: int | None = None
    arm_progress: float = 0.0
    spray: bool = False


@dataclass
class Fire:
    id: int
    x: float
    y: float
    anchor_node: int
    intensity: float = 100.0
    checked: bool = False
    unreachable: bool = False


@dataclass
class Obstacle:
    id: int
    x: float
    y: float


@dataclass
class Pedestrian:
    id: int
    x: float
    y: float
    speed: float
    emoji: str
    path: list[int] = field(default_factory=list)
    idx: int = 1
    waiting: bool = False
    wait_kind: str = ""
    state: str = "walking"


@dataclass
class NpcCar:
    """路网上的私家车：只负责交通，不参与消防任务，遇消防车要让行。"""

    id: int
    x: float
    y: float
    heading: float
    cruise: float
    color: str
    route: list[int] = field(default_factory=list)
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    wp_index: int = 0
    speed: float = 0.0
    state: str = "driving"          # driving / yield_intersection / yield_roadside
    side: float = 0.0               # 路侧避让时向路缘平移的距离（只影响绘制位置）
    target_node: int | None = None
    retry: float = 0.0              # 到达终点后重选路线的冷却

    @property
    def yielding(self) -> bool:
        return self.state != "driving"


class World:
    def __init__(self, auto_spawn: bool = True, seed: int | None = None):
        self.rng = random.Random(seed)
        self.city = CityMap()
        self.auto_spawn = auto_spawn
        self.time = 0.0
        self.truck = Truck(x=154.0, y=300.0, heading=math.pi / 2)
        self.fires: list[Fire] = []
        self.obstacles: list[Obstacle] = []
        self.pedestrians: list[Pedestrian] = []
        self.cars: list[NpcCar] = []
        self.events: list[dict] = []
        self.stats = {
            "fires_out": 0,
            "refills": 0,
            "repairs": 0,
            "obstacles_cleared": 0,
            "self_checks": 0,
            "ai_routes": 0,
            "ai_stops": 0,
        }
        self.ai_mission: dict | None = None
        self._next_fire_id = 1
        self._next_obstacle_id = 1
        self._next_ped_id = 1
        self._next_car_id = 1
        self._check_index = 0
        self._check_timer = 0.0
        self._repair_timer = 0.0
        self._fire_timer = 6.0
        self._obstacle_timer = 18.0
        self._fault_timer = 45.0
        self._car_timer = 2.0        # 第一辆私家车稍早出现，方便演示
        self.self_check = self._empty_self_check()
        self._log("系统启动，消防小车在待命点完成初始化", "info")
        for _ in range(10):
            self._spawn_pedestrian()
        if auto_spawn:
            for _ in range(CAR_MAX):
                self.spawn_car()

    # ------------------------------------------------------------------ 基础工具
    @staticmethod
    def _empty_self_check() -> dict:
        return {
            "running": False,
            "reason": None,
            "fire_id": None,
            "progress": 0.0,
            "passed": None,
            "started_at": None,
            "finished_at": None,
            "steps": [
                {"key": key, "name": name, "status": "pending", "detail": "等待检测"}
                for key, name in SELF_CHECK_ITEMS
            ],
        }

    def _log(self, text: str, level: str = "info") -> None:
        self.events.append({"time": round(self.time, 1), "text": text, "level": level})
        if len(self.events) > 80:
            self.events = self.events[-60:]

    @property
    def signal(self) -> str:
        """南北绿灯 / 东西绿灯，20 秒一个完整周期。"""
        return "NS" if self.time % 20 < 10 else "EW"

    @property
    def signal_remaining(self) -> float:
        return round(10 - (self.time % 10), 1)

    def _dist(self, x1, y1, x2, y2) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    # ------------------------------------------------------------------ 主循环
    def update(self, dt: float) -> None:
        dt = max(0.0, min(float(dt), 0.2))
        if dt == 0:
            return
        self.time += dt
        self._update_pedestrians(dt)
        self._update_self_check(dt)
        self._update_truck(dt)
        self._update_cars(dt)
        if self.auto_spawn:
            self._update_spawners(dt)

    def _update_spawners(self, dt: float) -> None:
        self._fire_timer -= dt
        if self._fire_timer <= 0:
            self.spawn_fire()
            self._fire_timer = self.rng.uniform(22, 34)
        self._obstacle_timer -= dt
        if self._obstacle_timer <= 0:
            if len(self.obstacles) < 4:
                traveling = self.truck.state in ("to_fire", "to_refill", "to_repair")
                self.spawn_obstacle(ahead=traveling and self.rng.random() < 0.6)
            self._obstacle_timer = self.rng.uniform(16, 28)
        self._fault_timer -= dt
        if self._fault_timer <= 0:
            if self.truck.fault is None:
                self.inject_fault()
            self._fault_timer = self.rng.uniform(40, 65)
        self._car_timer -= dt
        if self._car_timer <= 0:
            # 常驻：正常情况下四辆一直在路上，只有数量不足时才补一辆
            if len(self.cars) < CAR_MAX:
                self.spawn_car()
            self._car_timer = 3.0

    # ------------------------------------------------------------------ 车辆逻辑
    def _update_truck(self, dt: float) -> None:
        t = self.truck
        if t.state == "self_check":
            t.speed = 0.0
            return
        if t.state in ("to_fire", "to_refill", "to_repair", "ai_mission"):
            if t.state == "to_fire" and not self._fire_by_id(t.target_fire):
                self._enter_idle("目标火情已消失")
                return
            if t.state == "ai_mission":
                self._update_arm(dt)
                self._drive(dt, self._target_speed())
                self._check_ai_arrival()
                return
            self._update_arm(dt)
            target_speed = self._target_speed()
            self._drive(dt, target_speed)
            self._check_arrival()
            return
        t.speed = 0.0
        if t.state == "extinguishing":
            self._extinguish(dt)
        elif t.state == "refilling":
            self._refill(dt)
        elif t.state == "repairing":
            self._repair(dt)
        elif t.state == "ai_scan":
            self._ai_scan(dt)
        elif t.state == "idle":
            self._idle()

    def _drive(self, dt: float, target_speed: float) -> None:
        t = self.truck
        if not t.waypoints or t.wp_index >= len(t.waypoints):
            target_speed = 0.0

        # 接近目标点时转向；大角度转弯先减速，避免“漂移”
        if t.waypoints and t.wp_index < len(t.waypoints):
            tx, ty = t.waypoints[t.wp_index]
            dx, dy = tx - t.x, ty - t.y
            dist = math.hypot(dx, dy)
            if dist < 14:
                t.wp_index += 1
                if t.wp_index >= len(t.waypoints):
                    target_speed = 0.0
            elif dist > 0:
                desired = math.atan2(dy, dx)
                diff = (desired - t.heading + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) > 0.95:
                    target_speed = min(target_speed, 30.0)
                if t.speed > 3:
                    t.heading += max(-TURN_RATE * dt, min(TURN_RATE * dt, diff))

        if t.speed < target_speed:
            t.speed = min(target_speed, t.speed + ACCEL * dt)
        else:
            t.speed = max(target_speed, t.speed - ACCEL * 1.6 * dt)

        t.x += math.cos(t.heading) * t.speed * dt
        t.y += math.sin(t.heading) * t.speed * dt
        t.heading = (t.heading + math.pi) % (2 * math.pi) - math.pi
        t.x = max(10.0, min(self.city.WIDTH - 10.0, t.x))
        t.y = max(10.0, min(self.city.HEIGHT - 10.0, t.y))

    def _target_speed(self) -> float:
        t = self.truck
        target = BASE_SPEED
        t.slowing = False
        if t.arm_target is not None:
            return 0.0
        for p in self.pedestrians:
            dx, dy = p.x - t.x, p.y - t.y
            dist = math.hypot(dx, dy)
            if dist > 130 or dist < 0.1:
                continue
            diff = (math.atan2(dy, dx) - t.heading + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) < 0.62:                 # 正前方 ±35° 的锥形区域
                t.slowing = True
                target = min(target, 0.0 if dist < 58 else SLOW_SPEED)
        return target

    def _check_arrival(self) -> None:
        t = self.truck
        if t.state == "to_fire":
            fire = self._fire_by_id(t.target_fire)
            if fire is None:
                self._enter_idle("目标火情已消失")
                return
            ax, ay = self.city.road.xy[fire.anchor_node]
            if self._dist(t.x, t.y, ax, ay) <= 40:
                t.state = "extinguishing"
                t.spray = True
                t.speed = 0.0
                self._log(f"到达火场 #{fire.id}，开始喷水灭火", "info")
        elif t.state == "to_refill":
            st = self.city.stations[t.target_station] if t.target_station is not None else None
            if st:
                ax, ay = self.city.road.xy[st["node"]]
                if self._dist(t.x, t.y, ax, ay) <= 45:
                    t.state = "refilling"
                    t.speed = 0.0
                    self._log(f"抵达{st['name']}，开始补水", "info")
        elif t.state == "to_repair":
            st = self.city.stations[t.target_station] if t.target_station is not None else None
            if st:
                ax, ay = self.city.road.xy[st["node"]]
                if self._dist(t.x, t.y, ax, ay) <= 45:
                    t.state = "repairing"
                    t.speed = 0.0
                    self._repair_timer = 0.0
                    self._log(f"抵达{st['name']}，开始检修", "warning")

    def _enter_idle(self, reason: str = "") -> None:
        t = self.truck
        t.state = "idle"
        t.speed = 0.0
        t.spray = False
        t.route = []
        t.waypoints = []
        t.wp_index = 0
        t.target_fire = None
        t.target_station = None
        if reason:
            self._log(reason, "info")

    def _idle(self) -> None:
        t = self.truck
        if t.fault:
            if not self.self_check["running"]:
                self.start_self_check("fault")
            return
        if t.water < WATER_LOW:
            if self.dispatch_refill():
                return
        unchecked = [f for f in self.fires if not f.checked and not f.unreachable]
        if unchecked:
            self.start_self_check("fire", unchecked[0].id)
            return
        checked = [f for f in self.fires if f.checked and not f.unreachable]
        if checked:
            self.dispatch_fire(checked[0])

    def _extinguish(self, dt: float) -> None:
        t = self.truck
        fire = self._fire_by_id(t.target_fire)
        if fire is None:
            t.spray = False
            if self.ai_mission:
                self._advance_ai("目标火情已消失，继续 AI 路线")
            else:
                self._enter_idle()
            return
        if self._dist(t.x, t.y, fire.x, fire.y) > 100:
            t.spray = False
            if self.ai_mission:
                self._ai_goto(self.ai_mission["index"])     # 沿 AI 路线重新贴近火点
            else:
                t.state = "to_fire"
                self._plan_to_fire(fire)
            return
        t.spray = True
        t.water = max(0.0, t.water - SPRAY_WATER * dt)
        fire.intensity = max(0.0, fire.intensity - SPRAY_POWER * dt)
        if t.water <= 0:
            t.spray = False
            self._log("灭火剂耗尽，火情未完全扑灭，立即前往最近补水站", "critical")
            self._end_ai_mission("水量耗尽，AI 任务中断")
            self.dispatch_refill(resume_fire=fire.id)
            return
        if fire.intensity <= 0:
            self.fires = [f for f in self.fires if f.id != fire.id]
            self.stats["fires_out"] += 1
            t.spray = False
            self._log(f"火情 #{fire.id} 已扑灭（剩余水量 {t.water:.0f}%）", "success")
            if self.ai_mission:
                self._advance_ai(f"火情 #{fire.id} 已扑灭，AI 路线继续")
                return
            if t.water < WATER_LOW:
                self.dispatch_refill()
            else:
                self._enter_idle()

    def _refill(self, dt: float) -> None:
        t = self.truck
        t.water = min(100.0, t.water + REFILL_RATE * dt)
        t.battery = min(100.0, t.battery + 20.0 * dt)
        if t.water >= 100.0:
            t.water = 100.0
            self.stats["refills"] += 1
            self._log("补水完成，水量 100%", "success")
            self._enter_idle()

    def _repair(self, dt: float) -> None:
        self._repair_timer += dt
        if self._repair_timer >= REPAIR_TIME:
            t = self.truck
            old = t.fault
            t.fault = None
            self.stats["repairs"] += 1
            self._log(f"维修完成，{FAULT_NAMES.get(old or '', '故障')}已恢复正常", "success")
            self._enter_idle()

    # ------------------------------------------------------------------ 机械臂 / 障碍物
    def _update_arm(self, dt: float) -> None:
        t = self.truck
        if t.arm_target is not None:
            obstacle = next((o for o in self.obstacles if o.id == t.arm_target), None)
            if obstacle is None:
                t.arm_target = None
                t.arm_progress = 0.0
                return
            t.arm_progress = min(1.0, t.arm_progress + dt / 1.6)
            if t.arm_progress >= 1.0:
                self.obstacles = [o for o in self.obstacles if o.id != obstacle.id]
                self.stats["obstacles_cleared"] += 1
                self._log(f"机械臂已将路障 #{obstacle.id} 移出车道", "success")
                t.arm_target = None
                t.arm_progress = 0.0
            return
        nearest = None
        nearest_d = 9999.0
        for o in self.obstacles:
            dx, dy = o.x - t.x, o.y - t.y
            dist = math.hypot(dx, dy)
            if dist > 95 or dist < 0.1:
                continue
            diff = (math.atan2(dy, dx) - t.heading + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) < 0.5 and dist < nearest_d:
                nearest, nearest_d = o, dist
        if nearest is not None:
            t.arm_target = nearest.id
            t.arm_progress = 0.0
            self._log(f"前方发现路障 #{nearest.id}，停车使用机械臂清理", "warning")

    def _update_obstacles(self) -> None:
        # 障碍物本身是静态的；保留方法方便后续扩展（例如被火引燃）
        return

    # ------------------------------------------------------------------ 寻路 / 调度
    def _lane_waypoints(self, route: list[int]) -> list[tuple[float, float]]:
        """把道路节点序列转成“靠右车道”的连续航点。"""
        pts = [self.city.road.xy[n] for n in route]
        waypoints: list[tuple[float, float]] = []
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < 0.5:
                continue
            rx, ry = -dy / length, dx / length       # 行驶方向的右侧法向量
            waypoints.append((x2 + rx * LANE, y2 + ry * LANE))
        return waypoints

    def _plan_route(self, goal_node: int) -> tuple[list[int], float] | None:
        start_node = self.city.nearest_road_node(self.truck.x, self.truck.y)
        route = graph_astar(self.city.road.adj, start_node, goal_node, self.city.road.xy)
        if not route:
            return None
        length = 0.0
        for a, b in zip(route, route[1:]):
            (x1, y1), (x2, y2) = self.city.road.xy[a], self.city.road.xy[b]
            length += math.hypot(x2 - x1, y2 - y1)
        return route, length

    def _set_route(self, route: list[int], node: int) -> None:
        t = self.truck
        t.route = list(route)
        t.waypoints = self._lane_waypoints(route)
        t.wp_index = 0
        t.target_station = None
        t.target_fire = None
        # 终点节点就是目标停靠点；若航点为空，车辆已在目标点
        if not t.waypoints and node is not None:
            tx, ty = self.city.road.xy[node]
            t.waypoints = [(tx, ty)]

    def dispatch_fire(self, fire: Fire) -> bool:
        t = self.truck
        planned = self._plan_route(fire.anchor_node)
        if planned is None:
            self._log(f"无法规划到火情 #{fire.id} 的路线，请检查路障", "critical")
            fire.checked = True
            fire.unreachable = True
            return False
        route, _length = planned
        t.state = "to_fire"
        t.spray = False
        self._set_route(route, fire.anchor_node)
        t.target_fire = fire.id
        self._log(f"自检通过，出发前往火情 #{fire.id}", "info")
        fire.checked = True
        return True

    def _plan_to_fire(self, fire: Fire) -> None:
        planned = self._plan_route(fire.anchor_node)
        if planned:
            self._set_route(planned[0], fire.anchor_node)
            self.truck.target_fire = fire.id

    def _nearest_station(self, kind: str) -> tuple[int, list[int], float] | None:
        best = None
        for idx, st in enumerate(self.city.stations):
            if st["kind"] != kind:
                continue
            planned = self._plan_route(st["node"])
            if planned is None:
                continue
            if best is None or planned[1] < best[2]:
                best = (idx, planned[0], planned[1])
        return best

    def dispatch_refill(self, resume_fire: int | None = None) -> bool:
        t = self.truck
        if self.ai_mission:
            self._end_ai_mission("水量偏低，转入补水流程")
        best = self._nearest_station("refill")
        if best is None:
            self._log("所有补水站都不可达，等待人工处理", "critical")
            return False
        idx, route, _length = best
        st = self.city.stations[idx]
        t.state = "to_refill"
        t.target_station = idx
        t.target_fire = resume_fire
        t.spray = False
        self._set_route(route, st["node"])
        t.target_station = idx
        if resume_fire:
            t.target_fire = resume_fire
        self._log(f"水量偏低，前往最近补水站：{st['name']}", "warning")
        return True

    def dispatch_repair(self) -> bool:
        t = self.truck
        if self.ai_mission:
            self._end_ai_mission("车辆故障，转入维修流程")
        best = self._nearest_station("repair")
        if best is None:
            self._log("维修站不可达，等待人工处理", "critical")
            return False
        idx, route, _length = best
        st = self.city.stations[idx]
        t.state = "to_repair"
        t.target_station = idx
        t.target_fire = None
        t.spray = False
        self._set_route(route, st["node"])
        t.target_station = idx
        self._log(f"检测到{FAULT_NAMES.get(t.fault or '', '功能')}故障，前往{st['name']}", "critical")
        return True

    # ------------------------------------------------------------------ AI 最优路线
    def _ai_targets(self) -> list[dict]:
        """当前值得跑一趟的目标点：有火情就去火情，没有就做全域巡检。"""
        targets = []
        for fire in sorted(self.fires, key=lambda f: f.id):
            if fire.intensity <= 0:
                continue
            targets.append({
                "node": fire.anchor_node,
                "kind": "fire",
                "label": f"火情 #{fire.id}",
                "fire_id": fire.id,
                "intensity": round(fire.intensity),
            })
        if not targets:
            for i, spot in enumerate(self.city.fire_spots):
                targets.append({
                    "node": spot["node"],
                    "kind": "patrol",
                    "label": f"巡检点 {i + 1}",
                })
        return targets

    def plan_ai_route(self) -> dict:
        """只做规划，不改变车辆状态；供按钮点击和「执行」前的预演使用。"""
        t = self.truck
        if self.self_check["running"]:
            return {"ok": False, "reason": "全车自检进行中，稍后再规划路线"}
        start = self.city.nearest_road_node(t.x, t.y)
        targets = self._ai_targets()
        result = plan_multi_route(
            self.city.road.adj, self.city.road.xy, start, targets,
            start_point=(round(t.x, 1), round(t.y, 1)),
        )
        if not result.get("ok"):
            return result

        has_fire = any(x["kind"] == "fire" for x in targets)
        result["mode"] = "fire" if has_fire else "patrol"
        result["mode_label"] = "多火情最优处置路线" if has_fire else "全域预防性巡检路线"
        result["polyline"] = [[round(t.x, 1), round(t.y, 1)]] + [
            [round(x, 1), round(y, 1)]
            for x, y in self._lane_waypoints(result["path_nodes"])
        ]
        result["truck"] = {"x": round(t.x, 1), "y": round(t.y, 1), "state": t.state}
        result["explanation"] = [
            f"识别到 {result['algorithm']['targets']} 个目标点，先用 A* 算出两两最短路，"
            f"再用 {result['algorithm']['optimizer']} 决定访问顺序。",
            f"最优总里程 {result['total_distance']:.0f}，按发现顺序要跑 "
            f"{result['baseline_distance']:.0f}，节省 {result['saved_distance']:.0f}"
            f"（{result['saved_percent']:.1f}%）。",
            f"全程 {len(result['stops'])} 站，预计 {result['eta']:.0f} 秒跑完。",
        ]
        return result

    def execute_ai_route(self) -> dict:
        """接管车辆，按 AI 规划的顺序出车。"""
        t = self.truck
        if self.ai_mission:
            return {"ok": False, "msg": "AI 路线正在执行中"}
        if self.self_check["running"] or t.state == "self_check":
            return {"ok": False, "msg": "自检进行中，请稍后再接管"}
        plan = self.plan_ai_route()
        if not plan.get("ok"):
            return plan
        if not plan["stops"]:
            return {"ok": False, "msg": "没有可执行的目标点"}

        self.ai_mission = {
            "mode": plan["mode"],
            "plan": plan,
            "index": 0,
            "visited": [],
            "dwell": 0.0,
            "started_at": round(self.time, 1),
        }
        self.stats["ai_routes"] += 1
        t.state = "ai_mission"
        t.spray = False
        t.target_fire = None
        t.target_station = None
        self._log(f"AI 接管：{plan['mode_label']}，共 {len(plan['stops'])} 站，"
                  f"预计 {plan['eta']:.0f} 秒", "info")
        self._ai_goto(0)
        return {"ok": True, "mode": plan["mode"], "mode_label": plan["mode_label"],
                "stops": len(plan["stops"]), "total_distance": plan["total_distance"],
                "eta": plan["eta"], "algorithm": plan["algorithm"]["name"]}

    def _ai_goto(self, index: int) -> None:
        mission = self.ai_mission
        if not mission:
            return
        stops = mission["plan"]["stops"]
        if index >= len(stops):
            self._end_ai_mission("AI 路线全部完成")
            return
        mission["index"] = index
        mission["dwell"] = 0.0
        stop = stops[index]
        planned = self._plan_route(stop["node"])
        if planned is None:
            self._log(f"AI 无法到达 {stop['label']}，跳过该目标", "critical")
            self._advance_ai()
            return

        route, _length = planned
        t = self.truck
        t.state = "ai_mission"
        t.spray = False
        t.route = list(route)
        t.waypoints = self._lane_waypoints(route)
        t.wp_index = 0
        if not t.waypoints:
            tx, ty = self.city.road.xy[stop["node"]]
            t.waypoints = [(tx, ty)]
        t.target_fire = stop.get("fire_id") if stop["kind"] == "fire" else None
        t.target_station = None
        self._log(f"AI 导航：前往第 {index + 1}/{len(stops)} 站 {stop['label']}", "info")

    def _check_ai_arrival(self) -> None:
        mission = self.ai_mission
        if not mission:
            return
        stops = mission["plan"]["stops"]
        idx = mission["index"]
        if idx >= len(stops):
            self._end_ai_mission("AI 路线全部完成")
            return
        stop = stops[idx]
        ax, ay = self.city.road.xy[stop["node"]]
        t = self.truck
        if self._dist(t.x, t.y, ax, ay) > AI_ARRIVE:
            return
        if stop["kind"] == "fire":
            fire = self._fire_by_id(stop.get("fire_id"))
            if fire is None or fire.intensity <= 0:
                self._advance_ai(f"{stop['label']} 已无火情，继续前往下一站")
                return
            t.state = "extinguishing"
            t.speed = 0.0
            t.spray = True
            t.target_fire = fire.id
            self._log(f"到达 {stop['label']}，开始喷水灭火", "info")
        else:
            t.state = "ai_scan"
            t.speed = 0.0
            self._log(f"到达 {stop['label']}，正在扫描街区", "info")

    def _ai_scan(self, dt: float) -> None:
        mission = self.ai_mission
        if not mission:
            self._enter_idle()
            return
        mission["dwell"] += dt
        if mission["dwell"] >= AI_SCAN_TIME:
            idx = mission["index"]
            label = mission["plan"]["stops"][idx]["label"] if idx < len(mission["plan"]["stops"]) else "巡检点"
            self._advance_ai(f"{label} 巡检完成，未发现异常")

    def _advance_ai(self, message: str = "") -> None:
        mission = self.ai_mission
        if not mission:
            return
        if message:
            self._log(message, "info")
        idx = mission["index"]
        stops = mission["plan"]["stops"]
        if 0 <= idx < len(stops):
            mission["visited"].append(stops[idx]["order"])
            self.stats["ai_stops"] += 1
        self._ai_goto(idx + 1)

    def _end_ai_mission(self, reason: str = "") -> None:
        mission = self.ai_mission
        if not mission:
            return
        self.ai_mission = None
        done = len(mission["visited"])
        total = len(mission["plan"]["stops"])
        level = "success" if done >= total else "warning"
        self._log(f"AI 路线结束：{reason}（完成 {done}/{total} 站）", level)
        if self.truck.state in ("ai_mission", "ai_scan", "extinguishing"):
            self._enter_idle()

    def cancel_ai(self) -> dict:
        if not self.ai_mission:
            return {"ok": False, "msg": "当前没有 AI 任务"}
        self._end_ai_mission("手动取消")
        return {"ok": True}

    def ai_status(self) -> dict:
        """给前端的心跳字段：当前 AI 任务进度与绘制用的整条路线。"""
        mission = self.ai_mission
        if not mission:
            return {
                "active": False, "mode": None, "mode_label": None,
                "index": 0, "total": 0, "visited": [],
                "stops": [], "polyline": [],
                "fires": len(self.fires),
            }
        plan = mission["plan"]
        return {
            "active": True,
            "mode": plan["mode"],
            "mode_label": plan["mode_label"],
            "index": mission["index"],
            "total": len(plan["stops"]),
            "visited": list(mission["visited"]),
            "total_distance": plan["total_distance"],
            "eta": plan["eta"],
            "stops": plan["stops"],
            "polyline": plan["polyline"],
            "fires": len(self.fires),
        }

    # ------------------------------------------------------------------ 自检
    def start_self_check(self, reason: str = "manual", fire_id: int | None = None) -> dict:
        if self.self_check["running"]:
            return {"ok": False, "msg": "自检正在进行中"}
        if self.ai_mission:
            self._end_ai_mission("开始全车自检")
        t = self.truck
        t.state = "self_check"
        t.speed = 0.0
        t.spray = False
        t.route = []
        t.waypoints = []
        t.wp_index = 0
        t.arm_target = None
        t.arm_progress = 0.0
        self._check_index = 0
        self._check_timer = 0.0
        self.self_check = self._empty_self_check()
        self.self_check.update({
            "running": True,
            "reason": reason,
            "fire_id": fire_id,
            "started_at": round(self.time, 1),
            "progress": 0.0,
        })
        self.stats["self_checks"] += 1
        label = {"fire": "火情触发", "fault": "故障触发", "manual": "手动触发"}.get(reason, reason)
        self._log(f"启动全车自检（{label}）", "info")
        return {"ok": True}

    def _update_self_check(self, dt: float) -> None:
        sc = self.self_check
        if not sc["running"]:
            return
        if self._check_index >= len(sc["steps"]):
            self._finish_self_check()
            return
        step = sc["steps"][self._check_index]
        if step["status"] == "pending":
            step["status"] = "running"
            self._check_timer = 0.0
        self._check_timer += dt
        if self._check_timer >= SELF_CHECK_STEP:
            self._evaluate_step(step)
            self._check_index += 1
            self._check_timer = 0.0
            sc["progress"] = round(self._check_index / len(sc["steps"]), 3)
            if self._check_index >= len(sc["steps"]):
                self._finish_self_check()

    def _evaluate_step(self, step: dict) -> None:
        t = self.truck
        key = step["key"]
        ok = True
        if key == "water":
            ok = t.water >= WATER_LOW
            step["detail"] = f"水量 {t.water:.0f}% / 阈值 {WATER_LOW:.0f}%"
            if not ok:
                step["detail"] += "，需要补水"
        elif key == "battery":
            ok = t.battery >= 20
            step["detail"] = f"电量 {t.battery:.0f}%"
            if not ok:
                step["detail"] += "，需要充电"
        else:
            ok = t.fault != key
            step["detail"] = "工作正常" if ok else f"检测到{FAULT_NAMES[key]}故障"
        step["status"] = "pass" if ok else "fail"

    def _finish_self_check(self) -> None:
        sc = self.self_check
        if not sc["running"]:
            return
        passed = all(step["status"] == "pass" for step in sc["steps"])
        if self.truck.fault:
            passed = False
        sc["running"] = False
        sc["passed"] = passed
        sc["progress"] = 1.0
        sc["finished_at"] = round(self.time, 1)
        sc["steps"] = [dict(step) for step in sc["steps"]]
        if passed:
            self._log("自检全部通过，车辆状态正常", "success")
            reason, fire_id = sc["reason"], sc["fire_id"]
            if reason == "fire" and fire_id is not None:
                fire = self._fire_by_id(fire_id)
                if fire:
                    if not self.dispatch_fire(fire):
                        self._enter_idle()
                else:
                    self._enter_idle()
            else:
                self._enter_idle()
        else:
            failed = [s["name"] for s in sc["steps"] if s["status"] != "pass"]
            if not failed and self.truck.fault:
                failed = [FAULT_NAMES.get(self.truck.fault, "未知故障")]
            self._log("自检未通过：" + "、".join(failed), "critical")
            if self.truck.fault:
                self.dispatch_repair()
            elif self.truck.water < WATER_LOW:
                self.dispatch_refill()
            else:
                self._enter_idle()

    # ------------------------------------------------------------------ 故障 / 火情 / 障碍
    def inject_fault(self, kind: str | None = None) -> dict:
        if self.truck.fault:
            return {"ok": False, "msg": f"已存在故障：{FAULT_NAMES.get(self.truck.fault)}"}
        kind = kind or self.rng.choice(list(FAULT_NAMES))
        if kind not in FAULT_NAMES:
            return {"ok": False, "msg": "未知故障类型"}
        self.truck.fault = kind
        self._log(f"车辆突发故障：{FAULT_NAMES[kind]}", "critical")
        if self.self_check["running"]:
            # 自检过程中突发故障：直接把对应项目标记为失败，页面结果保持一致
            for step in self.self_check["steps"]:
                if step["key"] == kind:
                    step["status"] = "fail"
                    step["detail"] = f"检测到{FAULT_NAMES[kind]}故障"
        if not self.self_check["running"]:
            if self.truck.state == "idle":
                self.start_self_check("fault")
            elif self.truck.state not in ("to_repair", "repairing", "self_check"):
                self.dispatch_repair()
        return {"ok": True, "fault": kind}

    def spawn_fire(self) -> dict:
        used = {(round(f.x, 1), round(f.y, 1)) for f in self.fires}
        free = [s for s in self.city.fire_spots if (round(s["x"], 1), round(s["y"], 1)) not in used]
        if not free:
            return {"ok": False, "msg": "所有点位都已有火情"}
        spot = self.rng.choice(free)
        fire = Fire(id=self._next_fire_id, x=spot["x"], y=spot["y"],
                    anchor_node=spot["node"])
        self._next_fire_id += 1
        self.fires.append(fire)
        self._log(f"传感器检测到火情 #{fire.id}，立即启动自检", "critical")
        if self.truck.state == "idle" and not self.self_check["running"]:
            self.start_self_check("fire", fire.id)
        return {"ok": True, "fire_id": fire.id}

    def spawn_obstacle(self, ahead: bool = False) -> dict:
        if len(self.obstacles) >= 5:
            return {"ok": False, "msg": "路障太多了"}
        t = self.truck
        if ahead:
            x = t.x + math.cos(t.heading) * 120
            y = t.y + math.sin(t.heading) * 120
            x = max(30.0, min(self.city.WIDTH - 30.0, x))
            y = max(30.0, min(self.city.HEIGHT - 30.0, y))
        else:
            if self.rng.random() < 0.5:
                x = self.rng.choice(self.city.V_ROADS) + self.rng.choice((-LANE, LANE))
                y = self.rng.uniform(30, self.city.HEIGHT - 30)
            else:
                y = self.rng.choice(self.city.H_ROADS) + self.rng.choice((-LANE, LANE))
                x = self.rng.uniform(30, self.city.WIDTH - 30)
            if self._dist(x, y, t.x, t.y) < 120:
                return {"ok": False, "msg": "离小车太近，换个位置"}
        obstacle = Obstacle(id=self._next_obstacle_id, x=x, y=y)
        self._next_obstacle_id += 1
        self.obstacles.append(obstacle)
        self._log(f"道路上出现路障 #{obstacle.id}", "warning")
        return {"ok": True, "obstacle_id": obstacle.id}

    # ------------------------------------------------------------------ 行人
    def _spawn_pedestrian(self) -> None:
        for _attempt in range(8):
            start = self.city.random_ped_node(self.rng)
            goal = self.city.random_ped_node(self.rng)
            if start == goal:
                continue
            path = self.city.ped_route(start, goal)
            if len(path) < 2:
                continue
            x, y = self.city.ped.xy[start]
            emoji = self.rng.choice(["🚶", "🚶", "🚶‍♂️", "🚶‍♀️"])
            ped = Pedestrian(id=self._next_ped_id, x=x, y=y,
                             speed=self.rng.uniform(20, 30), emoji=emoji,
                             path=path, idx=1)
            self._next_ped_id += 1
            self._maybe_wait(ped)
            self.pedestrians.append(ped)
            return

    def _maybe_wait(self, ped: Pedestrian) -> None:
        if ped.idx >= len(ped.path):
            return
        kind = self.city.ped_edge_kind.get((ped.path[ped.idx - 1], ped.path[ped.idx]))
        if kind in ("cross_ew", "cross_ns") and not self._cross_allowed(kind):
            ped.waiting = True
            ped.wait_kind = kind
            ped.state = "waiting"
        else:
            ped.waiting = False
            ped.state = "crossing" if kind and kind.startswith("cross") else "walking"

    def _cross_allowed(self, kind: str) -> bool:
        if kind == "cross_ew":       # 东西向过街，和东西向车流同时放行
            return self.signal == "EW"
        if kind == "cross_ns":
            return self.signal == "NS"
        return True

    def _new_ped_route(self, ped: Pedestrian) -> None:
        goal = self.city.random_ped_node(self.rng)
        start = self.city.ped_nodes.get((round(ped.x, 1), round(ped.y, 1)))
        if start is None:
            start = self.city.random_ped_node(self.rng)
        path = self.city.ped_route(start, goal)
        if len(path) < 2:
            return
        ped.path = path
        ped.idx = 1
        self._maybe_wait(ped)

    def _update_pedestrians(self, dt: float) -> None:
        for ped in self.pedestrians:
            if ped.waiting:
                if self._cross_allowed(ped.wait_kind):
                    ped.waiting = False
                    ped.state = "crossing"
                else:
                    continue
            if ped.idx >= len(ped.path):
                self._new_ped_route(ped)
                if ped.idx >= len(ped.path):
                    continue
            tx, ty = self.city.ped.xy[ped.path[ped.idx]]
            dx, dy = tx - ped.x, ty - ped.y
            dist = math.hypot(dx, dy)
            if dist < 3:
                ped.x, ped.y = tx, ty
                ped.idx += 1
                if ped.idx >= len(ped.path):
                    self._new_ped_route(ped)
                else:
                    self._maybe_wait(ped)
                continue
            step = min(dist, ped.speed * dt)
            ped.x += dx / dist * step
            ped.y += dy / dist * step

    # ------------------------------------------------------------------ 私家车（NPC）
    def spawn_car(self) -> dict:
        """在路网随机位置放一辆私家车，路上常驻 CAR_MAX 辆。"""
        if len(self.cars) >= CAR_MAX:
            return {"ok": False, "msg": f"路上最多 {CAR_MAX} 辆小车"}
        nodes = self.city.road.xy
        fallback = None
        for _attempt in range(24):
            start = self.rng.randrange(len(nodes))
            goal = self.rng.randrange(len(nodes))
            if start == goal:
                continue
            sx, sy = nodes[start]
            if self._dist(sx, sy, self.truck.x, self.truck.y) < CAR_SPAWN_CLEAR:
                continue
            if any(self._dist(sx, sy, c.x, c.y) < 70 for c in self.cars):
                continue
            route = graph_astar(self.city.road.adj, start, goal, nodes)
            if len(route) < 3:
                continue
            waypoints = self._lane_waypoints(route)
            if not waypoints:
                continue
            length = sum(self._dist(*nodes[a], *nodes[b])
                         for a, b in zip(route, route[1:]))
            if length >= CAR_ROUTE_MIN:
                return self._add_car(start, route, waypoints)
            if fallback is None:
                fallback = (start, route, waypoints)
        if fallback is not None:
            return self._add_car(*fallback)
        return {"ok": False, "msg": "暂时没有合适的出生点"}

    def _add_car(self, start: int, route: list[int],
                 waypoints: list[tuple[float, float]]) -> dict:
        sx, sy = self.city.road.xy[start]
        car = NpcCar(
            id=self._next_car_id, x=sx, y=sy,
            heading=math.atan2(waypoints[0][1] - sy, waypoints[0][0] - sx),
            cruise=self.rng.uniform(CAR_SPEED_MIN, CAR_SPEED_MAX),
            color=self.rng.choice(CAR_COLORS),
            route=route, waypoints=waypoints, target_node=route[-1],
        )
        self._next_car_id += 1
        self.cars.append(car)
        self._log(f"小车 #{car.id} 上路（路上共 {len(self.cars)}/{CAR_MAX} 辆）", "info")
        return {"ok": True, "car_id": car.id}

    def _truck_busy(self) -> bool:
        """消防车是否正在出车途中；只有出车时私家车才需要让行。"""
        return self.truck.state in ("to_fire", "to_refill", "to_repair", "ai_mission")

    def _update_cars(self, dt: float) -> None:
        for car in self.cars:
            self._car_yield(car)
            self._car_drive(car, dt)
            if car.wp_index >= len(car.waypoints):
                # 常驻：到终点不停驶离，换一条随机路线继续跑
                car.speed = 0.0
                car.retry -= dt
                if car.retry <= 0:
                    car.retry = 1.0
                    self._car_new_route(car)
        self._resolve_overlaps()

    def _car_new_route(self, car: NpcCar) -> bool:
        """给常驻小车换一条新路线：从当前位置最近的节点出发，随机挑一个终点。"""
        nodes = self.city.road.xy
        start = self.city.nearest_road_node(car.x, car.y)
        for _attempt in range(10):
            goal = self.rng.randrange(len(nodes))
            if goal == start:
                continue
            route = graph_astar(self.city.road.adj, start, goal, nodes)
            if len(route) < 3:
                continue
            waypoints = self._lane_waypoints(route)
            if not waypoints:
                continue
            car.route = route
            car.waypoints = waypoints
            car.wp_index = 0
            car.target_node = goal
            return True
        return False

    def _truck_oncoming(self, car: NpcCar) -> bool:
        """消防车是不是迎面驶来（车头方向大致相反）。"""
        diff = (self.truck.heading - car.heading + math.pi) % (2 * math.pi) - math.pi
        return abs(diff) > 1.75

    def _car_drawn_xy(self, car: NpcCar) -> tuple[float, float]:
        """小车的实际绘制位置（车道位置 + 横向让行偏移）。"""
        return (car.x - math.sin(car.heading) * car.side,
                car.y + math.cos(car.heading) * car.side)

    def _truck_gap(self, car: NpcCar) -> tuple[float, float] | None:
        """消防车相对小车的（纵向距离, 横向偏移）；只算车头前方的车。"""
        cx, cy = self._car_drawn_xy(car)
        dx, dy = self.truck.x - cx, self.truck.y - cy
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return None
        rel = (math.atan2(dy, dx) - car.heading + math.pi) % (2 * math.pi) - math.pi
        longi = dist * math.cos(rel)
        if longi <= 0:
            return None
        return longi, abs(dist * math.sin(rel))

    def _car_yield(self, car: NpcCar) -> None:
        """让行判定：目标是「两车任何时候都不重叠」。

        - 消防车迎面开来 / 从后方追上来：小车靠到路肩停下，等它过去；
        - 同侧同向、消防车在前面因为补水 / 维修 / 灭火等原因停下：
          小车停到它后面等着，消防车重新起步后再跟着走；
        - 前方就是路口且消防车正在出车：停在路口外，让它先过；
        - 消防车在另一条路上（横向错开）：各走各的，不用让。
        """
        cx, cy = self._car_drawn_xy(car)
        dist = self._dist(cx, cy, self.truck.x, self.truck.y)
        limit = CAR_CLEAR_DIST if car.yielding else CAR_YIELD_DIST
        if dist > limit:
            car.state = "driving"
            return

        gap = self._truck_gap(car)
        if gap is None:
            # 消防车在侧后方：只有它还在出车、会追上来时才靠边让行
            car.state = "yield_roadside" if self._truck_busy() else "driving"
            return

        ahead_dist, side_dist = gap
        if side_dist > 36 or ahead_dist < 10:
            car.state = "driving"                  # 不在同一车道，互不相干
            return

        if self._truck_oncoming(car):
            car.state = "yield_roadside"           # 迎面来车：靠边停，让出车道
        elif self._truck_busy() and self._truck_in_intersection(car):
            car.state = "yield_intersection"       # 路口：停在路口外等它先过
        elif self.truck.speed < 5:
            car.state = "wait_truck"               # 同侧、消防车停下了：排队等它起步
        else:
            car.state = "driving"                  # 同侧、它还在走：跟着走

    def _truck_in_intersection(self, car: NpcCar) -> bool:
        """小车快到路口，而且消防车正接近 / 通过同一个路口。"""
        node = self._intersection_ahead(car)
        if node is None:
            return False
        nx, ny = self.city.road.xy[node]
        return self._dist(self.truck.x, self.truck.y, nx, ny) < 100.0

    def _intersection_ahead(self, car: NpcCar) -> int | None:
        """车辆正前方最近的路口；只有进入 CAR_INTERSECTION_DIST 才算「在路口」。"""
        for node in car.route[car.wp_index + 1:]:
            if not self.city.is_intersection(node):
                continue
            nx, ny = self.city.road.xy[node]
            if self._dist(car.x, car.y, nx, ny) <= CAR_INTERSECTION_DIST:
                return node
            return None                          # 最近的路口还远，后面的更不用看
        return None

    def _car_target_side(self, car: NpcCar) -> float:
        """让行时的横向目标：+ 靠路肩，其余情况走车道中心。"""
        return CAR_PULL_OVER if car.state == "yield_roadside" else 0.0

    def _car_target_speed(self, car: NpcCar) -> float:
        if car.state == "yield_intersection":
            return 0.0
        if car.state == "yield_roadside":
            # 先减速向路肩靠，贴边后停住
            return 0.0 if car.side >= CAR_PULL_OVER - 0.5 else min(car.cruise, 30.0)
        if car.state == "wait_truck":
            # 同侧前方消防车停下了：开到它后面停住，等它起步
            gap = self._truck_gap(car)
            if gap is None or gap[0] < CAR_WAIT_GAP:
                return 0.0
            return min(car.cruise, 28.0)
        gap = self._truck_gap(car)
        if gap is not None and gap[1] < 36 and gap[0] < 120:
            # 同侧前方是行驶中的消防车：按它的车速跟车，保持车距
            return min(car.cruise, max(self.truck.speed, 0.0))
        return car.cruise

    def _car_drive(self, car: NpcCar, dt: float) -> None:
        target_speed = min(self._car_target_speed(car), self._car_follow_speed(car))
        if car.wp_index < len(car.waypoints):
            tx, ty = car.waypoints[car.wp_index]
            dx, dy = tx - car.x, ty - car.y
            dist = math.hypot(dx, dy)
            if dist < 12:
                car.wp_index += 1
            elif dist > 0:
                desired = math.atan2(dy, dx)
                diff = (desired - car.heading + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) > 0.95:
                    target_speed = min(target_speed, 26.0)
                if car.speed > 3:
                    car.heading += max(-CAR_TURN * dt, min(CAR_TURN * dt, diff))
        else:
            target_speed = 0.0

        if car.speed < target_speed:
            car.speed = min(target_speed, car.speed + CAR_ACCEL * dt)
        else:
            car.speed = max(target_speed, car.speed - CAR_BRAKE * dt)

        car.x += math.cos(car.heading) * car.speed * dt
        car.y += math.sin(car.heading) * car.speed * dt
        car.heading = (car.heading + math.pi) % (2 * math.pi) - math.pi
        car.x = max(10.0, min(self.city.WIDTH - 10.0, car.x))
        car.y = max(10.0, min(self.city.HEIGHT - 10.0, car.y))

        # 横向让行：靠路肩（+）或借对向车道（-），让完自动回正
        want = self._car_target_side(car)
        step = CAR_SIDE_RATE * dt
        if car.side < want:
            car.side = min(want, car.side + step)
        elif car.side > want:
            car.side = max(want, car.side - step)

        # 重叠由 _update_cars 末尾的 _resolve_overlaps() 统一消解

    def _car_follow_speed(self, car: NpcCar) -> float:
        """跟车：同一车道前方有别的车，就按车距减速，避免追尾重叠。"""
        cap = car.cruise
        cx, cy = self._car_drawn_xy(car)
        for other in self.cars:
            if other.id == car.id:
                continue
            ox, oy = self._car_drawn_xy(other)
            dx, dy = ox - cx, oy - cy
            dist = math.hypot(dx, dy)
            if dist > 80 or dist < 0.1:
                continue
            rel = (math.atan2(dy, dx) - car.heading + math.pi) % (2 * math.pi) - math.pi
            if abs(dist * math.sin(rel)) > 26:
                continue                       # 不在同一车道，互不影响
            gap = dist * math.cos(rel)
            if gap <= 0:
                continue                       # 在自己后面
            if gap < CAR_CAR_GAP + 6:
                return 0.0                     # 贴得太近：停车
            if gap < 60:
                cap = min(cap, 22.0)
            else:
                cap = min(cap, max(other.speed, 24.0))
        return cap

    def _resolve_overlaps(self) -> None:
        """全局消解重叠：小车之间、小车与消防车之间都不允许重叠。

        每次迭代把所有过近的车对推开一点，几轮之后收敛；
        消防车是优先车辆，只推小车、不推它。
        """
        for _ in range(8):
            moved = False
            # 小车之间：各退一半
            for i in range(len(self.cars)):
                a = self.cars[i]
                ax, ay = self._car_drawn_xy(a)
                for j in range(i + 1, len(self.cars)):
                    b = self.cars[j]
                    bx, by = self._car_drawn_xy(b)
                    dx, dy = ax - bx, ay - by
                    d = math.hypot(dx, dy)
                    if d >= CAR_CAR_GAP:
                        continue
                    if d < 1e-3:
                        dx, dy, d = 1.0, 0.0, 1.0
                    push = (CAR_CAR_GAP - d) / 2.0
                    a.x += dx / d * push
                    a.y += dy / d * push
                    b.x -= dx / d * push
                    b.y -= dy / d * push
                    a.speed = min(a.speed, 12.0)
                    b.speed = min(b.speed, 12.0)
                    ax, ay = self._car_drawn_xy(a)
                    bx, by = self._car_drawn_xy(b)
                    moved = True
            # 小车与消防车：只把小车推开
            for car in self.cars:
                cx, cy = self._car_drawn_xy(car)
                dx, dy = cx - self.truck.x, cy - self.truck.y
                d = math.hypot(dx, dy)
                if d >= CAR_MIN_GAP:
                    continue
                if d < 1e-3:
                    dx, dy, d = 1.0, 0.0, 1.0
                push = CAR_MIN_GAP - d
                car.x += dx / d * push
                car.y += dy / d * push
                car.speed = min(car.speed, 12.0)
                moved = True
            if not moved:
                return
        # 兜底：别把车推到地图外
        for car in self.cars:
            car.x = max(10.0, min(self.city.WIDTH - 10.0, car.x))
            car.y = max(10.0, min(self.city.HEIGHT - 10.0, car.y))

    # ------------------------------------------------------------------ 查询
    def _fire_by_id(self, fire_id: int | None) -> Fire | None:
        if fire_id is None:
            return None
        return next((f for f in self.fires if f.id == fire_id), None)

    def get_state(self) -> dict:
        t = self.truck
        spray_target = None
        if t.spray:
            fire = self._fire_by_id(t.target_fire)
            if fire:
                spray_target = {"x": round(fire.x, 1), "y": round(fire.y, 1)}
        arm = None
        if t.arm_target is not None:
            obstacle = next((o for o in self.obstacles if o.id == t.arm_target), None)
            if obstacle:
                arm = {"active": True, "progress": round(t.arm_progress, 2),
                       "target": {"x": round(obstacle.x, 1), "y": round(obstacle.y, 1)}}
        return {
            "time": round(self.time, 1),
            "signal": self.signal,
            "signal_remaining": self.signal_remaining,
            "car_limit": CAR_MAX,
            "truck": {
                "x": round(t.x, 1), "y": round(t.y, 1),
                "heading": round(t.heading, 3),
                "speed": round(t.speed, 1),
                "water": round(t.water, 1),
                "battery": round(t.battery, 1),
                "state": t.state,
                "slowing": t.slowing,
                "fault": t.fault,
                "fault_name": FAULT_NAMES.get(t.fault or "", None),
                "spray": t.spray,
                "spray_target": spray_target,
                "arm": arm,
                "target_fire": t.target_fire,
            },
            "pedestrians": [
                {"id": p.id, "x": round(p.x, 1), "y": round(p.y, 1),
                 "emoji": p.emoji, "state": p.state}
                for p in self.pedestrians
            ],
            "cars": [
                {"id": c.id,
                 # 绘制坐标已含横向让行偏移
                 "x": round(self._car_drawn_xy(c)[0], 1),
                 "y": round(self._car_drawn_xy(c)[1], 1),
                 "heading": round(c.heading, 3),
                 "speed": round(c.speed, 1),
                 "state": c.state,
                 "color": c.color}
                for c in self.cars
            ],
            "fires": [
                {"id": f.id, "x": round(f.x, 1), "y": round(f.y, 1),
                 "intensity": round(f.intensity, 1)}
                for f in self.fires
            ],
            "obstacles": [{"id": o.id, "x": round(o.x, 1), "y": round(o.y, 1)}
                          for o in self.obstacles],
            "self_check": copy.deepcopy(self.self_check),
            "ai": self.ai_status(),
            "events": self.events[-14:][::-1],
            "stats": dict(self.stats),
        }

    def reset(self) -> None:
        self.__init__(auto_spawn=self.auto_spawn, seed=None)
        self._log("仿真已重置", "info")
