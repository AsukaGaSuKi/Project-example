"""私家车（NPC）测试：常驻数量、自动接续路线、路口让行、路侧避让与消防车优先。

约定：世界推进只走 world.update()，不启动后台线程，结果可复现。
"""
import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sim.world import (  # noqa: E402
    World, NpcCar, BASE_SPEED, CAR_MAX, CAR_PULL_OVER, CAR_COLORS,
)

N150 = (170.0, 150.0)
N265 = (170.0, 265.0)
N380 = (170.0, 380.0)


def make_world(seed=1):
    return World(auto_spawn=False, seed=seed)


def advance(world, seconds, step=0.1):
    for _ in range(int(seconds / step)):
        world.update(step)


def node(world, xy):
    return world.city.road_nodes[xy]


def add_car(world, x, y, heading, waypoints, route_keys=(N150, N265, N380)):
    """放一辆位置和路线都可控的私家车，方便验证让行规则。"""
    car = NpcCar(
        id=world._next_car_id, x=x, y=y, heading=heading, cruise=60.0, color="#c8102e",
        route=[node(world, k) for k in route_keys],
        waypoints=list(waypoints),
        target_node=node(world, route_keys[-1]),
    )
    world._next_car_id += 1
    world.cars.append(car)
    return car


def dispatch_truck(world, x, y):
    """把消防车摆成「正在出车」的状态（ai_mission 不需要真的规划就能推进）。"""
    world.truck.state = "ai_mission"
    world.truck.x, world.truck.y = x, y
    world.truck.waypoints = [(x, y)]
    world.truck.wp_index = 0


# ---------------------------------------------------------------- 常驻规则
def test_启动就常驻四辆小车():
    w = World(auto_spawn=True, seed=11)
    assert len(w.cars) == CAR_MAX == 4
    assert w.spawn_car()["ok"] is False        # 满了就不再生成


def test_小车常驻不消失():
    w = World(auto_spawn=True, seed=12)
    ids = sorted(c.id for c in w.cars)
    for _ in range(900):                      # 90 秒仿真
        w.update(0.1)
        assert len(w.cars) == CAR_MAX         # 数量不变
    assert sorted(c.id for c in w.cars) == ids   # 还是那四辆，没有重建
    assert w._next_car_id - 1 == CAR_MAX      # 没有额外生成


def test_小车抵达终点后自动接下一段路():
    w = make_world()
    car = add_car(w, 154.0, 300.0, math.pi / 2, [(154.0, 305.0)])
    w.update(0.1)
    assert any(c.id == car.id for c in w.cars)     # 没有驶离路网
    assert car.waypoints and car.wp_index == 0     # 换了新路线，继续跑
    advance(w, 1.0)
    assert car.speed > 0.0


def test_小车统一使用消防红的深色():
    w = World(auto_spawn=True, seed=13)
    for car in w.cars:
        assert car.color in CAR_COLORS
        r, g, b = (int(car.color[i:i + 2], 16) for i in (1, 3, 5))
        assert r > 130 and g < 70 and b < 80       # 深红：红分量高，绿蓝很低


# ---------------------------------------------------------------- 路口让行
def test_在路口遇消防车_小车停车让行():
    w = make_world()
    # 车头前方 70 单位就是路口 (170, 380)
    car = add_car(w, 154.0, 310.0, math.pi / 2,
                  [(154.0, 380.0), (154.0, 500.0)], route_keys=(N150, N380, N265))
    dispatch_truck(w, 154.0, 420.0)
    advance(w, 1.0)

    assert car.state == "yield_intersection"
    assert car.speed == 0.0
    # 停在路口外，没有开进交叉口
    assert car.y < 380 - 36


def test_路口让行时消防车优先通过():
    w = make_world()
    add_car(w, 154.0, 310.0, math.pi / 2,
            [(154.0, 380.0), (154.0, 500.0)], route_keys=(N150, N380, N265))
    dispatch_truck(w, 154.0, 250.0)
    w.truck.waypoints = [(154.0, 250.0), (154.0, 320.0), (154.0, 460.0)]
    w.truck.wp_index = 0
    advance(w, 1.0)
    assert w.truck.speed == BASE_SPEED        # 消防车没有被拦下
    assert w.truck.slowing is False


# ---------------------------------------------------------------- 路侧避让
def test_非路口遇消防车_小车靠边避让():
    w = make_world()
    # 150 到 265 之间，前方最近的路口 (380) 还远，不算「在路口」
    car = add_car(w, 154.0, 215.0, math.pi / 2,
                  [(154.0, 265.0), (154.0, 380.0)], route_keys=(N150, N265, N380))
    dispatch_truck(w, 120.0, 300.0)
    advance(w, 2.5)

    assert car.state == "yield_roadside"
    assert car.speed == 0.0
    assert car.side > 0.5                     # 已经向路缘平移
    assert car.side <= CAR_PULL_OVER + 0.01


def test_消防车通过后小车恢复行驶():
    w = make_world()
    car = add_car(w, 154.0, 215.0, math.pi / 2,
                  [(154.0, 265.0), (154.0, 380.0)], route_keys=(N150, N265, N380))
    dispatch_truck(w, 120.0, 300.0)
    advance(w, 2.5)
    assert car.state == "yield_roadside"

    w.truck.x, w.truck.y = 900.0, 700.0       # 消防车开走
    advance(w, 3.0)
    assert car.state == "driving"
    assert car.speed > 10.0
    assert car.side < 0.01                    # 回到车道中心


def test_停着的消防车不会卡住小车():
    w = make_world()
    car = add_car(w, 154.0, 215.0, math.pi / 2,
                  [(154.0, 265.0), (154.0, 380.0)], route_keys=(N150, N265, N380))
    w.truck.state = "idle"
    w.truck.x, w.truck.y = 120.0, 300.0       # 就近停着，但没有出车
    advance(w, 1.0)
    assert car.state == "driving"
    assert car.speed > 0.0


# ---------------------------------------------------------------- 状态导出
def test_状态里带上四辆小车的绘制字段():
    w = World(auto_spawn=True, seed=14)
    cars = w.get_state()["cars"]
    assert len(cars) == CAR_MAX
    for c in cars:
        assert {"id", "x", "y", "heading", "speed", "state", "color"} <= set(c)
        assert c["state"] == "driving"
