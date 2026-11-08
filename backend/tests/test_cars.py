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
TRUCK_HALF = (15.0, 11.0)       # 消防车车身半长 / 半宽（已缩到和小车差不多）


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
def test_启动就常驻两辆小车():
    w = World(auto_spawn=True, seed=11)
    assert len(w.cars) == CAR_MAX == 2
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
# ---------------------------------------------------------------- 让行规则
def test_遇到消防车_小车靠边停车():
    w = make_world()
    car = add_car(w, 154.0, 310.0, math.pi / 2,
                  [(154.0, 380.0), (154.0, 500.0)], route_keys=(N150, N380, N265))
    dispatch_truck(w, 154.0, 390.0)            # 前方有消防车
    advance(w, 2.0)
    assert car.state == "yield_roadside"
    assert car.side > 0.5
    assert car.speed < 1.0


def test_让行时消防车优先通过():
    w = make_world()
    w.pedestrians = []                          # 排除行人降速干扰
    car = add_car(w, 154.0, 310.0, math.pi / 2,
                  [(154.0, 380.0), (154.0, 500.0)], route_keys=(N150, N380, N265))
    w.truck.state = "ai_mission"
    w.truck.x, w.truck.y = 90.0, 380.0
    w.truck.heading = 0.0
    w.truck.waypoints = [(300.0, 380.0)]
    w.truck.wp_index = 0
    advance(w, 1.0)
    assert car.state == "yield_roadside"
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


def test_同侧前方消防车停下_小车靠边等待():
    w = make_world()
    car = add_car(w, 154.0, 215.0, math.pi / 2,
                  [(154.0, 265.0), (154.0, 380.0)], route_keys=(N150, N265, N380))
    park_truck(w, 154.0, 300.0, state="idle")
    advance(w, 4.0)
    assert car.state == "yield_roadside"
    assert car.speed == 0.0
    assert car.y < 300.0


def test_消防车重新起步后小车跟上():
    w = make_world()
    car = add_car(w, 154.0, 215.0, math.pi / 2,
                  [(154.0, 265.0), (154.0, 380.0)], route_keys=(N150, N265, N380))
    park_truck(w, 154.0, 300.0, state="idle")
    advance(w, 4.0)
    assert car.state == "yield_roadside"

    w.truck.x, w.truck.y = 900.0, 700.0
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


# ---------------------------------------------------------------- 行人与会车
def test_小车会为前方行人停车():
    w = make_world()
    park_truck(w, 900.0, 700.0, state="idle")
    car = add_car(w, 154.0, 300.0, math.pi / 2, [(154.0, 380.0), (154.0, 500.0)])
    ped = w.pedestrians[0]
    ped.x, ped.y = 154.0, 352.0                # 车头前方 52 单位
    ped.speed = 0.0                            # 站着不动，方便验证
    w.pedestrians = [ped]
    advance(w, 2.0)
    assert car.speed < 6.0                      # 减速 / 停住，让人先过

    ped.x, ped.y = 900.0, 700.0                 # 行人走开
    advance(w, 2.0)
    assert car.speed > 10.0


def test_小车不会撞到行人():
    w = World(auto_spawn=True, seed=71)
    closest = 1e9
    for _ in range(900):                        # 90 秒
        w.update(0.1)
        for car in w.cars:
            cx, cy = drawn(w, car)
            for ped in w.pedestrians:
                closest = min(closest, math.hypot(ped.x - cx, ped.y - cy))
    assert closest >= 20.0, closest


def test_对向会车不会迎面相撞():
    w = make_world()
    a = add_car(w, 154.0, 320.0, math.pi / 2, [(154.0, 380.0)], route_keys=(N150, N380))
    b = add_car(w, 154.0, 250.0, -math.pi / 2, [(154.0, 150.0)], route_keys=(N380, N150))
    park_truck(w, 900.0, 700.0, state="idle")
    for _ in range(30):                          # 3 秒
        w.update(0.1)
        assert not body_overlaps(w)
    assert math.hypot(*(p - q for p, q in zip(drawn(w, a), drawn(w, b)))) >= 24.0


# ---------------------------------------------------------------- 拥堵疏通
def _jam_scene(truck_xy=(520.0, 380.0)):
    """在 (170, 380) 路口摆两辆停住的小车 + 一个行人，并放一辆近处的消防车。"""
    w = make_world()
    n150 = w.city.road_nodes[(170.0, 150.0)]
    n380 = w.city.road_nodes[(170.0, 380.0)]
    for cid, (x, y) in enumerate(((154.0, 352.0), (186.0, 356.0)), start=1):
        car = NpcCar(id=cid, x=x, y=y, heading=math.pi / 2, cruise=0.0,
                     color="#c8102e", route=[n150, n380],
                     waypoints=[(154.0, 500.0)], target_node=n380)
        w.cars.append(car)
    w._next_car_id = 3
    w.truck.state = "idle"
    w.truck.x, w.truck.y = truck_xy
    w.truck.heading = math.pi
    w.truck.waypoints = []
    w.truck.wp_index = 0
    w.pedestrians[0].x, w.pedestrians[0].y = 168.0, 372.0
    w.pedestrians[0].speed = 26.0
    return w, n380


def test_路口堵住会触发三步疏通():
    w, node = _jam_scene()
    advance(w, 6.0)
    assert w.jam is not None
    assert w.jam["node"] == node
    assert w.jam["phase"] in ("cars", "truck")
    assert w.stats["jam_clears"] == 1


def test_疏通时先疏散行人():
    w, node = _jam_scene()
    nx, ny = w.city.road.xy[node]
    ped = w.pedestrians[0]
    before = math.hypot(ped.x - nx, ped.y - ny)
    advance(w, 5.0)
    assert w.jam is not None and w.jam["phase"] in ("cars", "truck")
    after = math.hypot(ped.x - nx, ped.y - ny)
    assert after > before                       # 行人正往远离路口的方向走
    gx, gy = w.city.ped.xy[ped.path[-1]]
    assert math.hypot(gx - nx, gy - ny) > 120   # 新路线终点远离路口


def test_疏通时小车靠边让行():
    w, node = _jam_scene()
    advance(w, 6.0)
    assert w.jam is not None and w.jam["phase"] in ("cars", "truck")
    nx, ny = w.city.road.xy[node]
    near = [c for c in w.cars
            if math.hypot(*[a - b for a, b in zip(drawn(w, c), (nx, ny))]) < 145]
    assert near
    assert any(c.pull_aside > 0 or c.side > 0.5 for c in near)   # 靠边了


def test_拥堵时不强制消防车前往():
    w, node = _jam_scene(truck_xy=(380.0, 380.0))
    for _ in range(150):                            # 15 秒
        w.update(0.1)
        assert w.truck.state not in ("to_jam", "clearing_jam")
    assert w.stats["jam_clears"] >= 1               # 拥堵处理确实跑过
    assert w.truck.state in ("idle", "self_check", "to_fire", "extinguishing")
    assert w.stats["jam_clears"] >= 1


# ---------------------------------------------------------------- 消防车被堵 / 故障
def test_消防车被堵_先让小车撤离():
    w = make_world()
    w.cars = []
    n150 = w.city.road_nodes[(170.0, 150.0)]
    n380 = w.city.road_nodes[(170.0, 380.0)]
    for cid, (x, y) in enumerate(((154.0, 330.0), (186.0, 336.0)), start=1):
        w.cars.append(NpcCar(id=cid, x=x, y=y, heading=math.pi / 2, cruise=0.0,
                             color="#c8102e", route=[n150, n380],
                             waypoints=[(154.0, 500.0)], target_node=n380))
    w._next_car_id = 3
    w.truck.state = "ai_mission"
    w.truck.x, w.truck.y = 170.0, 300.0
    w.truck.heading = math.pi / 2
    w.truck.waypoints = [(170.0, 500.0)]
    w.truck.wp_index = 0

    advance(w, 4.0)
    assert w.clear_way is not None
    assert "堵" in w.clear_way["reason"]
    assert all(c.escape > 0 for c in w.cars)     # 小车进入撤离状态


def test_消防车在车流中故障_先让小车走再回维修站():
    w = make_world()
    w.cars = []
    n150 = w.city.road_nodes[(170.0, 150.0)]
    n380 = w.city.road_nodes[(170.0, 380.0)]
    w.cars.append(NpcCar(id=1, x=170.0, y=330.0, heading=math.pi / 2, cruise=0.0,
                         color="#c8102e", route=[n150, n380],
                         waypoints=[(154.0, 500.0)], target_node=n380))
    w._next_car_id = 2
    w.truck.x, w.truck.y = 170.0, 300.0
    w.inject_fault("pump")

    advance(w, 1.0)
    assert w.clear_way is not None
    assert w.cars[0].escape > 0
    advance(w, 4.0)
    assert w.clear_way is None
    assert w.truck.state in ("to_repair", "repairing", "self_check")
    assert w.truck.fault == "pump"


def test_小车不会开进建筑():
    for seed in (81, 82):
        w = World(auto_spawn=True, seed=seed)
        bad = 0
        for _ in range(900):                      # 90 秒
            w.update(0.1)
            for car in w.cars:
                x, y = drawn(w, car)
                if w._in_building(x, y, margin=0):
                    bad += 1
        assert bad == 0, (seed, bad)


def test_堵住时允许小车掉头换路():
    w = make_world()
    car = add_car(w, 154.0, 320.0, -math.pi / 2, [(154.0, 150.0)])
    # 车头朝北，正常路线都在身后；允许掉头后一定能拿到一条新路线
    assert w._car_new_route(car, blocked=(), uturn=True)
    assert car.waypoints


def test_消防车与小车堵住_小车朝反方向驶离():
    w = make_world()
    w.cars = []
    n150 = w.city.road_nodes[(170.0, 150.0)]
    n380 = w.city.road_nodes[(170.0, 380.0)]
    car = NpcCar(id=1, x=170.0, y=340.0, heading=math.pi / 2, cruise=50.0,
                 color="#c8102e", route=[n150, n380],
                 waypoints=[(154.0, 500.0)], target_node=n380)
    w.cars.append(car)
    w._next_car_id = 2
    w.truck.x, w.truck.y = 170.0, 300.0        # 小车在消防车正南
    w.truck.heading = math.pi / 2

    w._start_clear_way("消防车被小车堵住")
    assert car.escape > 0
    assert car.waypoints
    tx, ty = car.waypoints[0]
    dx, dy = tx - car.x, ty - car.y
    ax, ay = car.x - w.truck.x, car.y - w.truck.y     # 背离消防车的方向
    cos = (dx * ax + dy * ay) / (math.hypot(dx, dy) * math.hypot(ax, ay))
    assert cos > 0.2                                    # 目标点确实在反方向


def test_停稳的小车不会横在路面上():
    for seed in (81, 82):
        w = World(auto_spawn=True, seed=seed)
        hold = {}
        parked = bad = 0
        for _ in range(900):                       # 90 秒
            w.update(0.1)
            for car in w.cars:
                hold[car.id] = hold.get(car.id, 0.0) + 0.1 if abs(car.speed) <= 3 else 0.0
                if hold[car.id] < 0.5:
                    continue                        # 只是路过刹一下，不算停稳
                lane = w._car_lane_heading(car)
                if lane is None or car.side >= 30:
                    continue                        # 路边让行停靠允许斜停
                parked += 1
                diff = abs((lane - car.heading + math.pi) % (2 * math.pi) - math.pi)
                if min(diff, math.pi - diff) > math.pi / 4:   # 真正横过来（和车道线垂直）
                    bad += 1
        assert parked > 50
        assert bad / parked < 0.10, (seed, bad, parked)


# ---------------------------------------------------------------- 状态导出
def test_状态里带上两辆小车的绘制字段():
    w = World(auto_spawn=True, seed=14)
    cars = w.get_state()["cars"]
    assert len(cars) == CAR_MAX
    for c in cars:
        assert {"id", "x", "y", "heading", "speed", "state", "color"} <= set(c)
    assert w.get_state()["car_limit"] == CAR_MAX