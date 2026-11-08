"""AI 最优路线在仿真内核里的行为测试：规划、接管、逐站完成、取消。"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sim.world import World, Fire  # noqa: E402


def make_world(seed=1):
    w = World(auto_spawn=False, seed=seed)
    w.pedestrians = []          # 去掉行人，避免减速等待影响计时断言
    return w


def add_fire(w, index, fire_id, intensity=60.0):
    spot = w.city.fire_spots[index]
    fire = Fire(id=fire_id, x=spot["x"], y=spot["y"],
                anchor_node=spot["node"], intensity=intensity, checked=True)
    w.fires.append(fire)
    return fire


def advance(w, seconds, step=0.05):
    for _ in range(int(seconds / step)):
        w.update(step)


def test_有火情时规划为火情路线并覆盖所有火点():
    w = make_world(seed=11)
    add_fire(w, 0, 1)
    add_fire(w, 5, 2)
    add_fire(w, 9, 3)

    plan = w.plan_ai_route()
    assert plan["ok"] is True
    assert plan["mode"] == "fire"
    assert len(plan["stops"]) == 3
    assert {s["fire_id"] for s in plan["stops"]} == {1, 2, 3}
    assert plan["total_distance"] > 0
    assert plan["saved_distance"] >= 0
    assert len(plan["polyline"]) > 2
    assert plan["explanation"] and "A*" in plan["explanation"][0]


def test_没有火情时规划全域巡检路线():
    w = make_world(seed=12)
    plan = w.plan_ai_route()
    assert plan["ok"] is True
    assert plan["mode"] == "patrol"
    assert len(plan["stops"]) == len(w.city.fire_spots)
    assert plan["algorithm"]["targets"] == len(w.city.fire_spots)


def test_执行AI路线会自动跑完并扑灭全部火情():
    w = make_world(seed=13)
    add_fire(w, 0, 1, intensity=48)
    add_fire(w, 6, 2, intensity=48)

    result = w.execute_ai_route()
    assert result["ok"] is True
    assert w.truck.state == "ai_mission"
    assert w.ai_mission is not None

    advance(w, 150)
    assert w.fires == []
    assert w.stats["fires_out"] == 2
    assert w.ai_mission is None
    assert w.truck.state == "idle"
    assert w.stats["ai_stops"] >= 2
    assert w.truck.water < 100


def test_取消AI路线后恢复待命():
    w = make_world(seed=14)
    add_fire(w, 0, 1)
    assert w.execute_ai_route()["ok"] is True
    assert w.cancel_ai()["ok"] is True
    assert w.ai_mission is None
    assert w.truck.state == "idle"
    assert w.cancel_ai()["ok"] is False


def test_自检中拒绝AI接管():
    w = make_world(seed=15)
    add_fire(w, 0, 1)
    w.start_self_check("manual")
    result = w.execute_ai_route()
    assert result["ok"] is False
    assert w.ai_mission is None


def test_心跳状态包含AI进度字段():
    w = make_world(seed=16)
    add_fire(w, 0, 1)
    add_fire(w, 3, 2)
    w.execute_ai_route()
    state = w.get_state()
    ai = state["ai"]
    assert ai["active"] is True
    assert ai["total"] == 2
    assert ai["index"] == 0
    assert ai["mode"] == "fire"
    assert len(ai["polyline"]) > 2
    assert state["truck"]["state"] == "ai_mission"
