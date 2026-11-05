"""detector.py —— 告警规则

输入 world.get_state() 的一帧状态，输出页面告警列表。
规则保持简单、可解释，方便课程演示和单元测试。
"""

STATE_NAMES = {
    "idle": "待命",
    "self_check": "自检中",
    "to_fire": "赶赴火场",
    "extinguishing": "灭火中",
    "to_refill": "前往补水",
    "refilling": "补水/充电中",
    "to_repair": "前往维修",
    "repairing": "维修中",
}


def check(state):
    alerts = []
    truck = state["truck"]
    water = truck["water"]

    if water <= 0:
        alerts.append({"level": "critical", "type": "水量耗尽，已自动前往补水站"})
    elif water < 10:
        alerts.append({"level": "critical", "type": f"水量低于 10%（{water:.0f}%），需要补水"})
    elif water < 25:
        alerts.append({"level": "warning", "type": f"水量偏低（{water:.0f}%）"})

    if truck.get("fault"):
        name = truck.get("fault_name") or truck["fault"]
        if truck["state"] in ("to_repair", "repairing"):
            alerts.append({"level": "critical", "type": f"{name}故障，正在前往维修站"})
        else:
            alerts.append({"level": "critical", "type": f"{name}故障，将前往维修站"})

    for fire in state.get("fires", []):
        alerts.append({"level": "critical", "type": f"火情 #{fire['id']} 燃烧中（强度 {fire['intensity']:.0f}）"})

    if truck.get("slowing"):
        alerts.append({"level": "warning", "type": "前方有行人，自动降速慢行"})

    arm = truck.get("arm")
    if arm and arm.get("active"):
        alerts.append({"level": "warning", "type": "机械臂正在清理路障"})

    sc = state.get("self_check") or {}
    if sc.get("running"):
        alerts.append({"level": "info", "type": f"全车自检进行中（{int(sc.get('progress', 0) * 100)}%）"})
    elif sc.get("passed") is False:
        alerts.append({"level": "critical", "type": "最近一次自检未通过"})

    if not alerts:
        name = STATE_NAMES.get(truck["state"], truck["state"])
        alerts.append({"level": "info", "type": f"一切正常（{name}）"})
    return alerts
