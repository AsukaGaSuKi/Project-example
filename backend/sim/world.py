"""world.py —— 自动消防小车仿真内核

一台带机械臂的消防小车，在连续坐标的城市路网中完成：
- 火情检测 → 立刻自检（结果供前端可视化）→ 出发灭火；
- 水量低于 10% 自动前往最近补水站；灭火时允许用到 0% 再补水；
- 功能故障自动前往最近维修站；
- 行驶中遇到路障，停车用机械臂清障；
- 遇到行人自动降速慢行，行人通过后恢复；
- 行人随机生成，走人行道、过斑马线并遵守红绿灯。

所有状态由后端统一维护，update(dt) 是唯一推进仿真时间的入口，便于测试。
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

from algo.pathfinding import graph_astar
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
        self.events: list[dict] = []
        self.stats = {
            "fires_out": 0,
            "refills": 0,
            "repairs": 0,
            "obstacles_cleared": 0,
            "self_checks": 0,
        }
        self._next_fire_id = 1
        self._next_obstacle_id = 1
        self._next_ped_id = 1
        self._check_index = 0
        self._check_timer = 0.0
        self._repair_timer = 0.0
        self._fire_timer = 6.0
        self._obstacle_timer = 18.0
        self._fault_timer = 45.0
        self.self_check = self._empty_self_check()
        self._log("系统启动，消防小车在待命点完成初始化", "info")
        for _ in range(10):
            self._spawn_pedestrian()

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

    # ------------------------------------------------------------------ 车辆逻辑
    def _update_truck(self, dt: float) -> None:
        t = self.truck
        if t.state == "self_check":
            t.speed = 0.0
            return
        if t.state in ("to_fire", "to_refill", "to_repair"):
            if t.state == "to_fire" and not self._fire_by_id(t.target_fire):
                self._enter_idle("目标火情已消失")
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
            self._enter_idle()
            return
        if self._dist(t.x, t.y, fire.x, fire.y) > 100:
            t.spray = False
            t.state = "to_fire"
            self._plan_to_fire(fire)
            return
        t.spray = True
        t.water = max(0.0, t.water - SPRAY_WATER * dt)
        fire.intensity = max(0.0, fire.intensity - SPRAY_POWER * dt)
        if t.water <= 0:
            t.spray = False
            self._log("灭火剂耗尽，火情未完全扑灭，立即前往最近补水站", "critical")
            self.dispatch_refill(resume_fire=fire.id)
            return
        if fire.intensity <= 0:
            self.fires = [f for f in self.fires if f.id != fire.id]
            self.stats["fires_out"] += 1
            t.spray = False
            self._log(f"火情 #{fire.id} 已扑灭（剩余水量 {t.water:.0f}%）", "success")
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

    # ------------------------------------------------------------------ 自检
    def start_self_check(self, reason: str = "manual", fire_id: int | None = None) -> dict:
        if self.self_check["running"]:
            return {"ok": False, "msg": "自检正在进行中"}
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
            "fires": [
                {"id": f.id, "x": round(f.x, 1), "y": round(f.y, 1),
                 "intensity": round(f.intensity, 1)}
                for f in self.fires
            ],
            "obstacles": [{"id": o.id, "x": round(o.x, 1), "y": round(o.y, 1)}
                          for o in self.obstacles],
            "self_check": copy.deepcopy(self.self_check),
            "events": self.events[-14:][::-1],
            "stats": dict(self.stats),
        }

    def reset(self) -> None:
        self.__init__(auto_spawn=self.auto_spawn, seed=None)
        self._log("仿真已重置", "info")
