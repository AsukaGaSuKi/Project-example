"""私家车（NPC）测试：常驻、让行规则，以及「车身绝不重叠、也不会被挤开」。

重叠判定用凸多边形分离轴（SAT）算真实车身：小车 26×14、消防车约 42×30。
"""
import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sim.world import (  # noqa: E402
    World, NpcCar, BASE_SPEED, CAR_MAX, CAR_PULL_OVER, CAR_COLORS,
    CAR_CAR_LEN, CAR_CAR_SIDE, CAR_WAIT_GAP,
)

N150 = (170.0, 150.0)
N265 = (170.0, 265.0)
N380 = (170.0, 380.0)

CAR_HALF = (13.0, 7.0)          # 小车车身半长 / 半宽
TRUCK_HALF = (21.0, 15.0)       # 消防车车身半长 / 半宽


def make_world(seed=1):
    return World(auto_spawn=False, seed=seed)


def advance(world, seconds, step=0.1):
    for _ in range(int(seconds / step)):
        world.update(step)


def node(world, xy):
    return world.city.road_nodes[xy]


def add_car(world, x, y, heading, waypoints, route_keys=(N150, N265, N380)):
    car = NpcCar(
        id=world._next_car_id, x=x, y=y, heading=heading, cruise=60.0, color="#c8102e",
        route=[node(world, k) for k in route_keys],
        waypoints=list(waypoints),
        target_node=node(world, route_keys[-1]),
    )
    world._next_car_id += 1
    world.cars.append(car)
    return car


def dispatch_truck(world, x, y, heading=math.pi / 2):
    world.truck.state = "ai_mission"
    world.truck.x, world.truck.y = x, y
    world.truck.heading = heading
    world.truck.waypoints = [(x, y)]
    world.truck.wp_index = 0


def park_truck(world, x, y, state="idle", heading=math.pi / 2):
    world.truck.state = state
    world.truck.x, world.truck.y = x, y
    world.truck.heading = heading
    world.truck.speed = 0.0
    world.truck.waypoints = []
    world.truck.wp_index = 0


def drawn(world, car):
    return world._car_drawn_xy(car)


def _corners(x, y, heading, half):
    c, s = math.cos(heading), math.sin(heading)
    hl, hw = half
    return [(x + dx * c - dy * s, y + dx * s + dy * c)
            for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]


def _overlap(p1, p2):
    """分离轴定理：两块凸四边形是否相交。"""
    for poly in (p1, p2):
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            ax, ay = -(y2 - y1), x2 - x1
            n = math.hypot(ax, ay) or 1.0
            ax, ay = ax / n, ay / n
            a = [ax * px + ay * py for px, py in p1]
            b = [ax * px + ay * py for px, py in p2]
            if max(a) < min(b) or max(b) < min(a):
                return False
    return True


def body_overlaps(world):
    """当前这一帧有没有任何两辆车（含消防车）车身相交。"""
    boxes = [(c.id, _corners(*drawn(world, c), c.heading, CAR_HALF)) for c in world.cars]
    truck = _corners(world.truck.x, world.truck.y, world.truck.heading, TRUCK_HALF)
    for cid, box in boxes:
        if _overlap(box, truck):
            return True
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _overlap(boxes[i][1], boxes[j][1]):
                return True
    return False


# ---------------------------------------------------------------- 常驻规则
def test_启动就常驻四辆小车():
    w = World(auto_spawn=True, seed=11)
    assert len(w.cars) == CAR_MAX == 4
    assert w.spawn_car()["ok"] is False


def test_小车常驻不消失():
    w = World(auto_spawn=True, seed=12)
    ids = sorted(c.id for c in w.cars)
    for _ in range(900):                      # 90 秒
        w.update(0.1)
        assert len(w.cars) == CAR_MAX
    assert sorted(c.id for c in w.cars) == ids
    assert w._next_car_id - 1 == CAR_MAX


def test_小车抵达终点后自动接下一段路():
    w = make_world()
    car = add_car(w, 154.0, 300.0, math.pi / 2, [(154.0, 305.0)])
    w.update(0.1)
    assert any(c.id == car.id for c in w.cars)
    assert car.waypoints and car.wp_index == 0
    advance(w, 1.0)
    assert car.speed > 0.0


def test_小车统一使用消防红的深色():
    w = World(auto_spawn=True, seed=13)
    for car in w.cars:
        assert car.color in CAR_COLORS
        r, g, b = (int(car.color[i:i + 2], 16) for i in (1, 3, 5))
        assert r > 130 and g < 70 and b < 80


# ---------------------------------------------------------------- 让行规则
def test_路口遇消防车_小车停车让行():
    w = make_world()
    car = add_car(w, 154.0, 310.0, math.pi / 2,
                  [(154.0, 380.0), (154.0, 500.0)], route_keys=(N150, N380, N265))
    dispatch_truck(w, 154.0, 390.0)            # 正在通过前方路口
    advance(w, 1.0)
    assert car.state == "yield_intersection"
    assert car.speed == 0.0


def test_路口让行时消防车优先通过():
    w = make_world()
    w.pedestrians = []                          # 排除行人降速干扰
    # 小车在纵向路口外等，消防车横穿同一个路口：小车停住，消防车满速通过
    car = add_car(w, 154.0, 310.0, math.pi / 2,
                  [(154.0, 380.0), (154.0, 500.0)], route_keys=(N150, N380, N265))
    w.truck.state = "ai_mission"
    w.truck.x, w.truck.y = 90.0, 380.0
    w.truck.heading = 0.0
    w.truck.waypoints = [(300.0, 380.0)]
    w.truck.wp_index = 0
    advance(w, 1.0)
    assert car.state == "yield_intersection"
    assert w.truck.speed == BASE_SPEED
    assert w.truck.slowing is False


def test_迎面遇消防车_小车靠边停车():
    w = make_world()
    car = add_car(w, 154.0, 215.0, math.pi / 2,
                  [(154.0, 265.0), (154.0, 380.0)], route_keys=(N150, N265, N380))
    dispatch_truck(w, 120.0, 300.0, heading=-math.pi / 2)
    advance(w, 2.5)
    assert car.state == "yield_roadside"
    assert car.speed == 0.0
    assert car.side > 0.5
    assert car.side <= CAR_PULL_OVER + 0.01


def test_消防车通过后小车恢复行驶():
    w = make_world()
    car = add_car(w, 154.0, 215.0, math.pi / 2,
                  [(154.0, 265.0), (154.0, 380.0)], route_keys=(N150, N265, N380))
    dispatch_truck(w, 120.0, 300.0, heading=-math.pi / 2)
    advance(w, 2.5)
    assert car.state == "yield_roadside"

    w.truck.x, w.truck.y = 900.0, 700.0
    advance(w, 3.0)
    assert car.state == "driving"
    assert car.speed > 10.0
    assert car.side < 0.01


def test_同侧前方消防车停下_小车停车等待():
    w = make_world()
    car = add_car(w, 154.0, 215.0, math.pi / 2,
                  [(154.0, 265.0), (154.0, 380.0)], route_keys=(N150, N265, N380))
    park_truck(w, 154.0, 300.0, state="idle")
    advance(w, 4.0)
    assert car.state == "wait_truck"
    assert car.speed == 0.0
    assert car.y < 300.0
    assert 300.0 - car.y <= CAR_WAIT_GAP + 40


def test_消防车重新起步后小车跟上():
    w = make_world()
    car = add_car(w, 154.0, 215.0, math.pi / 2,
                  [(154.0, 265.0), (154.0, 380.0)], route_keys=(N150, N265, N380))
    park_truck(w, 154.0, 300.0, state="idle")
    advance(w, 4.0)
    assert car.state == "wait_truck"

    dispatch_truck(w, 154.0, 300.0)
    w.truck.waypoints = [(154.0, 300.0), (154.0, 500.0), (154.0, 620.0)]
    w.truck.wp_index = 0
    saw_driving = False
    for _ in range(60):
        w.update(0.1)
        if car.state == "driving" and car.speed > 10.0:
            saw_driving = True
    assert saw_driving
    assert car.y > 260.0


# ---------------------------------------------------------------- 不重叠
def test_补水维修时小车在消防车后面等待且不重叠():
    for state in ("refilling", "repairing"):
        w = make_world()
        car = add_car(w, 154.0, 180.0, math.pi / 2,
                      [(154.0, 265.0), (154.0, 380.0)], route_keys=(N150, N265, N380))
        park_truck(w, 170.0, 265.0, state=state)
        if state == "refilling":
            w.truck.water = 40.0
        bad = 0
        for _ in range(120):
            w.update(0.1)
            if body_overlaps(w):
                bad += 1
        assert bad == 0, (state, bad)
        assert car.y < 265.0


def test_任何时刻车身都不重叠():
    for seed in (61, 62):
        w = World(auto_spawn=True, seed=seed)
        bad = 0
        for _ in range(1200):                 # 120 秒
            w.update(0.1)
            if body_overlaps(w):
                bad += 1
        assert bad == 0, (seed, bad)


def test_没有挤开_单帧位移不会突变():
    w = World(auto_spawn=True, seed=63)
    prev = {c.id: drawn(w, c) for c in w.cars}
    max_jump = 0.0
    for _ in range(1200):
        w.update(0.1)
        for c in w.cars:
            x, y = drawn(w, c)
            px, py = prev[c.id]
            max_jump = max(max_jump, math.hypot(x - px, y - py))
            prev[c.id] = (x, y)
    # 最高车速 68，一个 0.1 秒步长最多走 6.8；超过 12 就说明有瞬移/推挤
    assert max_jump < 12.0, max_jump


def test_跟车会减速避免追尾():
    w = make_world()
    front = add_car(w, 154.0, 265.0, math.pi / 2,
                    [(154.0, 380.0), (154.0, 500.0)], route_keys=(N150, N380, N265))
    rear = add_car(w, 154.0, 215.0, math.pi / 2,
                   [(154.0, 265.0), (154.0, 380.0)], route_keys=(N150, N265, N380))
    park_truck(w, 900.0, 700.0, state="idle")
    front.cruise = 0.0
    front.speed = 0.0
    rear.speed = 55.0
    advance(w, 2.0)
    assert rear.speed < 30.0
    fx, fy = drawn(w, front)
    rx, ry = drawn(w, rear)
    assert math.hypot(fx - rx, fy - ry) >= min(CAR_CAR_LEN, CAR_CAR_SIDE)


# ---------------------------------------------------------------- 状态导出
def test_状态里带上四辆小车的绘制字段():
    w = World(auto_spawn=True, seed=14)
    cars = w.get_state()["cars"]
    assert len(cars) == CAR_MAX
    for c in cars:
        assert {"id", "x", "y", "heading", "speed", "state", "color"} <= set(c)
    assert w.get_state()["car_limit"] == CAR_MAX
