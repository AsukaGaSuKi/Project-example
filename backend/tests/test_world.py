"""仿真规则测试：全部在后台线程之外直接调用 world.update()，结果可复现。"""
import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sim.world import (  # noqa: E402
    World, Fire, Pedestrian, SLOW_SPEED, WATER_LOW, BASE_SPEED,
)


def make_world(seed=1):
    return World(auto_spawn=False, seed=seed)


def advance(world, seconds, step=0.1):
    for _ in range(int(seconds / step)):
        world.update(step)


def test_水量低于10_自动前往最近补水站():
    w = make_world()
    w.truck.water = 6
    w.update(0.1)
    assert w.truck.state == "to_refill"
    assert w.truck.target_station is not None
    station = w.city.stations[w.truck.target_station]
    assert station["kind"] == "refill"
    # 起点在西区，最近的是 0 号补水站
    assert w.truck.target_station == 0


def test_补水完成后恢复待命():
    w = make_world()
    w.truck.water = 6
    w.update(0.1)
    advance(w, 120)
    assert w.truck.state == "idle"
    assert w.truck.water == 100
    assert w.stats["refills"] == 1


def test_火情触发自检并可完成灭火():
    w = make_world(seed=3)
    w.spawn_fire()
    assert w.truck.state == "self_check"
    assert w.self_check["running"] is True
    advance(w, 2.5)
    assert w.self_check["passed"] is True
    assert all(step["status"] == "pass" for step in w.self_check["steps"])
    assert w.truck.state in ("to_fire", "extinguishing")
    advance(w, 90)
    assert w.stats["fires_out"] == 1
    assert w.fires == []
    assert w.truck.water < 100


def test_自检发现故障后前往维修站():
    w = make_world(seed=4)
    result = w.inject_fault("pump")
    assert result["ok"] is True
    assert w.truck.fault == "pump"
    advance(w, 3.0)
    assert w.self_check["passed"] is False
    assert w.truck.state == "to_repair"


def test_维修完成后清除故障():
    w = make_world(seed=5)
    w.inject_fault("arm")
    advance(w, 3.0)
    # 直接放到维修站门口，验证维修计时和恢复
    st = next(s for s in w.city.stations if s["kind"] == "repair")
    w.truck.x, w.truck.y = st["anchor"]
    w.truck.waypoints = []
    advance(w, 0.2)
    assert w.truck.state == "repairing"
    advance(w, 4.5)
    assert w.truck.state == "idle"
    assert w.truck.fault is None
    assert w.stats["repairs"] == 1


def test_灭火时水量可低于10_耗尽后才补水():
    w = make_world(seed=6)
    spot = w.city.fire_spots[3]
    fire = Fire(id=7, x=spot["x"], y=spot["y"], anchor_node=spot["node"], checked=True)
    w.fires = [fire]
    w.truck.state = "extinguishing"
    w.truck.target_fire = fire.id
    w.truck.x, w.truck.y = spot["anchor"]
    w.truck.water = 8

    w.update(0.1)
    assert w.truck.state == "extinguishing"       # 低于 10% 仍可继续灭火
    assert w.truck.water < WATER_LOW

    w.truck.water = 0.2
    w.update(0.1)
    assert w.truck.state == "to_refill"            # 耗尽才去补水
    assert w.fires and w.fires[0].id == fire.id    # 火没有被错误地扑灭


def test_机械臂清除前方路障():
    w = make_world(seed=7)
    spot = w.city.fire_spots[11]
    fire = Fire(id=9, x=spot["x"], y=spot["y"], anchor_node=spot["node"])
    w.fires = [fire]
    w.dispatch_fire(fire)
    w.truck.x, w.truck.y = 154, 300
    w.truck.heading = math.pi / 2
    w.truck.waypoints = [(154, 380), (154, 520)]
    w.truck.wp_index = 0
    w.spawn_obstacle(ahead=True)

    advance(w, 4.0)
    assert w.stats["obstacles_cleared"] == 1
    assert w.obstacles == []


def test_遇到行人自动降速():
    w = make_world(seed=8)
    spot = w.city.fire_spots[11]
    fire = Fire(id=10, x=spot["x"], y=spot["y"], anchor_node=spot["node"])
    w.fires = [fire]
    w.dispatch_fire(fire)
    w.truck.x, w.truck.y = 154, 300
    w.truck.heading = math.pi / 2
    w.truck.waypoints = [(154, 380), (154, 560)]
    w.truck.wp_index = 0
    w.truck.speed = BASE_SPEED

    ped = w.pedestrians[0]
    ped.x, ped.y = 154, 370
    advance(w, 0.6)
    assert w.truck.slowing is True
    assert w.truck.speed <= SLOW_SPEED + 1

    ped.x, ped.y = 700, 700
    advance(w, 2.0)
    assert w.truck.slowing is False
    assert w.truck.speed > SLOW_SPEED


def test_行人红灯在斑马线前等待_绿灯通行():
    w = make_world(seed=9)
    w.time = 0.0                                   # 南北绿灯：东西向过街应等待
    city = w.city
    a = city.ped_nodes[(123.0, 103.0)]
    b = city.ped_nodes[(217.0, 103.0)]
    ped = Pedestrian(id=99, x=123.0, y=103.0, speed=25.0, emoji="🚶", path=[a, b], idx=1)
    w._maybe_wait(ped)
    assert ped.waiting is True
    assert ped.state == "waiting"

    w.time = 12.0                                  # 进入东西绿灯
    w.pedestrians = [ped]
    w.update(0.1)
    assert ped.waiting is False
    assert ped.x > 123.0


def test_自检流程包含六个可视化项目():
    w = make_world(seed=10)
    w.start_self_check("manual")
    assert len(w.self_check["steps"]) == 6
    advance(w, 0.5)
    running = [s for s in w.self_check["steps"] if s["status"] == "running"]
    assert running and running[0]["name"]
    advance(w, 2.0)
    assert w.self_check["running"] is False
    assert w.self_check["passed"] is True
    assert w.self_check["progress"] == 1.0

def _ped_edges(world):
    edges = []
    for a, nbrs in world.city.ped.adj.items():
        for b, _ in nbrs:
            if a < b:
                edges.append((world.city.ped.xy[a], world.city.ped.xy[b]))
    return edges


def _dist_to_seg(p, a, b):
    (px, py), (x1, y1), (x2, y2) = p, a, b
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    if l2 == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def test_行人只在人行道和斑马线上走():
    w = make_world(seed=21)
    edges = _ped_edges(w)
    for _ in range(600):                       # 60 秒
        w.update(0.1)
        for ped in w.pedestrians:
            best = min(_dist_to_seg((ped.x, ped.y), a, b) for a, b in edges)
            assert best < 3.0, best


def test_行人过街只走斑马线():
    from sim.city import CityMap
    w = make_world(seed=22)
    road_in = 0
    for _ in range(600):
        w.update(0.1)
        for ped in w.pedestrians:
            kind = ""
            if 0 < ped.idx < len(ped.path):
                kind = w.city.ped_edge_kind.get((ped.path[ped.idx - 1], ped.path[ped.idx]), "")
            if kind in ("cross_ew", "cross_ns"):
                continue                        # 正在走斑马线，允许在车行道上
            dx = min(abs(ped.x - v) for v in CityMap.V_ROADS)
            dy = min(abs(ped.y - h) for h in CityMap.H_ROADS)
            if min(dx, dy) < 36:
                road_in += 1
    assert road_in == 0