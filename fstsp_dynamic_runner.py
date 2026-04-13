# -*- coding: utf-8 -*-
import copy
import time
import pandas as pd
import fstsp_solver
import fstsp_evaluator
import simulation as sim
import utils as ut
from dynamic_logic import split_route_by_decision_time, add_virtual_truck_position_node

def _uniq_triplets(seq):
    out = []
    seen = set()
    for t in (seq or []):
        try:
            tt = (int(t[0]), int(t[1]), int(t[2]))
        except Exception:
            continue
        if tt not in seen:
            seen.add(tt)
            out.append(tt)
    return out


def fstsp_quick_filter(data_prelim, full_route_cur, full_triplets_cur, decisions_raw, ab_cfg):
    """FSTSP 专属的快速过滤器（修复：不再深拷贝，接收物理护栏的 decisions_raw）"""
    res_work = fstsp_evaluator.evaluate_fstsp_system(
        data_prelim, full_route_cur, full_triplets_cur,
        truck_speed_units=sim.TRUCK_SPEED_UNITS, drone_speed_units=sim.DRONE_SPEED_UNITS,
        truck_road_factor=sim.TRUCK_ROAD_FACTOR,
        alpha_drone=float(ab_cfg.get("alpha_drone", 0.3)),
        lambda_late=float(ab_cfg.get("lambda_late", 50.0))
    )
    base_cost = res_work["cost"]
    base_late = res_work["total_late"]

    data_work = data_prelim  # 修复：不再使用 copy.deepcopy，直接在原对象上微调测试
    decisions_filtered = []

    qf_cost_max = float(ab_cfg.get("qf_cost_max", 30.0))
    qf_late_max = float(ab_cfg.get("qf_late_max", 0.5))

    # 遍历物理护栏初步审核过的 decisions
    for (cid, dec, nx, ny, reason) in decisions_raw:
        if dec != "ACCEPT":
            # 如果物理护栏已经拒绝了（比如已经路过、超出范围），直接继承拒绝结果
            decisions_filtered.append((cid, dec, nx, ny, reason))
            continue

        # 记录原坐标，用于回滚
        old_x, old_y = data_work.nodes[cid]["x"], data_work.nodes[cid]["y"]

        # 假装接受，修改坐标进行评估
        data_work.nodes[cid]["x"], data_work.nodes[cid]["y"] = nx, ny

        res_trial = fstsp_evaluator.evaluate_fstsp_system(
            data_work, full_route_cur, full_triplets_cur,
            truck_speed_units=sim.TRUCK_SPEED_UNITS, drone_speed_units=sim.DRONE_SPEED_UNITS,
            truck_road_factor=sim.TRUCK_ROAD_FACTOR,
            alpha_drone=float(ab_cfg.get("alpha_drone", 0.3)),
            lambda_late=float(ab_cfg.get("lambda_late", 50.0))
        )

        d_cost = res_trial["cost"] - base_cost
        d_late = res_trial["total_late"] - base_late

        if d_cost > qf_cost_max or d_late > qf_late_max:
            decisions_filtered.append((cid, "REJECT", nx, ny, f"FSTSP护栏拦截：Δcost={d_cost:.2f}, Δlate={d_late:.2f}"))
            # 拒绝则还原坐标
            data_work.nodes[cid]["x"], data_work.nodes[cid]["y"] = old_x, old_y
        else:
            base_cost = res_trial["cost"]
            base_late = res_trial["total_late"]
            decisions_filtered.append((cid, "ACCEPT", nx, ny, "满足FSTSP扰动容忍度"))

    return data_work, decisions_filtered
def run_fstsp_epoch(decision_time, t_prev, scene_idx, data_cur, full_route_cur, full_triplets_cur,
                    full_arrival_cur, offline_groups, nodeid2idx, ab_cfg, seed, verbose=False):
    t_start_total = time.time()

    # 0. 先对当前状态进行一次精密评估，拿到准确的节点完成时间 (finish_times)
    eval_cur = fstsp_evaluator.evaluate_fstsp_system(
        data_cur, full_route_cur, full_triplets_cur,
        truck_speed_units=sim.TRUCK_SPEED_UNITS,
        drone_speed_units=sim.DRONE_SPEED_UNITS,
        truck_road_factor=sim.TRUCK_ROAD_FACTOR,
        alpha_drone=0.3,
        lambda_late=float(ab_cfg.get("lambda_late", 50.0))
    )
    arrival_times = eval_cur.get("arrival_times", {})
    finish_times = eval_cur.get("finish_times", {})
    depart_times = eval_cur.get("depart_times", eval_cur.get("depart", {}))
    triplet_times = eval_cur.get("triplet_times", {})

    # 1. 提取物理前缀路线与虚拟坐标点
    served_nodes, _, current_node, virtual_pos, prefix_route = split_route_by_decision_time(
        full_route_cur, arrival_times, decision_time, data_cur.central_idx, data_cur
    )

    # 🚨 核心修复：基于精确的物理完成时间划分 unserved / served 客户
    unserved_customers = set()
    served_customers = set()
    for c in range(len(data_cur.nodes)):
        if str(data_cur.nodes[c].get('node_type', '')).lower() == 'customer':
            if finish_times.get(c, float('inf')) <= decision_time + 1e-9:
                served_customers.add(c)
            else:
                unserved_customers.add(c)

    # 提取本轮事件
    key = round(float(decision_time), 6)
    evs = offline_groups.get(key, [])
    req_override = []
    predefined_xy = {}

    for e in evs:
        nid = int(e.get('NODE_ID', 0))
        cidx = nodeid2idx.get(nid, None)
        if cidx is not None and cidx in unserved_customers:
            req_override.append(int(cidx))
            predefined_xy[int(cidx)] = (float(e.get('NEW_X', 0.0)), float(e.get('NEW_Y', 0.0)))

    # 调用 FSTSP 专属快滤网（修复：加入物理护栏）
    if req_override:
        from dynamic_logic import apply_relocations_for_decision_time, build_client_to_base_map

        # 1. 构造 client_to_base_cur (FSTSP 需要把 triplets 转成 dict 格式供护栏检查)
        b2d_cur = {}
        for l_id, c_id, r_id in full_triplets_cur:
            b2d_cur.setdefault(l_id, []).append(c_id)
        client_to_base_cur = build_client_to_base_map(b2d_cur)

        # 2. 先过物理规则（拦截已服务、拦截超范围、更新预选基站）
        data_prelim, decisions_raw, req_clients = apply_relocations_for_decision_time(
            data_cur, t_prev, decision_time,
            depart_times, finish_times, arrival_times, client_to_base_cur,
            req_override, predefined_xy, {}, {}
        )

        # 3. 再过 FSTSP 成本滤网
        data_next, decisions = fstsp_quick_filter(
            data_prelim, full_route_cur, full_triplets_cur, decisions_raw, ab_cfg
        )
    else:
        data_next, decisions = data_cur, []

    num_acc = sum(1 for d in decisions if str(d[1]).startswith("ACCEPT"))
    active_triplets = []
    for tri, tt in triplet_times.items():
        try:
            t_launch = float(tt["launch_time"])
            t_recover = float(tt["recover_time"])
        except Exception:
            continue
        if t_launch <= decision_time + 1e-9 and decision_time < t_recover - 1e-9:
            active_triplets.append(tri)
    prefix_triplets = []
    for t in full_triplets_cur:
        tt = triplet_times.get(t)
        # 最小修补：只有物理回收时间明确早于决策时间，才算作真正的历史前缀 Triplet
        if tt and tt.get("recover_time", float('inf')) <= decision_time + 1e-9:
            prefix_triplets.append(t)

    # 早停跳过
    if num_acc == 0 and len(req_override) == 0:
        return {
            'break': False, 'skip': True,
            'data_next': data_next, 'full_route_next': full_route_cur, 'full_triplets_next': full_triplets_cur,
            'full_arrival_next': arrival_times, 'full_finish_next': finish_times, 'full_depart_next': depart_times,
            'stat_record': ut._pack_scene_record(scene_idx, decision_time, eval_cur, num_req=len(req_override),
                                                 num_acc=num_acc, num_rej=len(req_override) - num_acc, alpha_drone=0.3,
                                                 lambda_late=50.0, solver_time=0),
            'decision_log_rows': [],
            'full_eval': eval_cur,
            'viz_pack': {
                'data': data_next,
                'route': full_route_cur,
                'triplets': full_triplets_cur,
                'decisions': decisions,
                'virtual_pos': virtual_pos,
                'prefix_route': prefix_route
            }
        }
    if active_triplets:
        full_eval_hold = fstsp_evaluator.evaluate_fstsp_system(
            data_next, full_route_cur, full_triplets_cur,
            truck_speed_units=sim.TRUCK_SPEED_UNITS,
            drone_speed_units=sim.DRONE_SPEED_UNITS,
            truck_road_factor=sim.TRUCK_ROAD_FACTOR,
            alpha_drone=0.3,
            lambda_late=float(ab_cfg.get("lambda_late", 50.0))
        )
        return {
            'break': False,
            'skip': True,
            'data_next': data_next,
            'full_route_next': full_route_cur,
            'full_triplets_next': full_triplets_cur,
            'full_arrival_next': full_eval_hold.get('arrival_times', {}),
            'full_finish_next': full_eval_hold.get('finish_times', {}),
            'full_depart_next': full_eval_hold.get('depart_times', full_eval_hold.get('depart', {})),
            'stat_record': ut._pack_scene_record(
                scene_idx, decision_time, full_eval_hold,
                num_req=len(req_override),
                num_acc=num_acc,
                num_rej=len(req_override) - num_acc,
                alpha_drone=0.3,
                lambda_late=50.0,
                solver_time=0
            ),
            'decision_log_rows': [],
            'full_eval': full_eval_hold,
            'viz_pack': {
                'data': data_next,
                'route': full_route_cur,
                'triplets': full_triplets_cur,
                'decisions': decisions,
                'virtual_pos': virtual_pos,
                'prefix_route': prefix_route
            }
        }
    # 构造 FSTSP 残局 DataFrame (基于 unserved_customers)
    start_idx_for_alns = add_virtual_truck_position_node(data_next, virtual_pos) if virtual_pos else current_node

    rows_fstsp = []
    nd_start = data_next.nodes[start_idx_for_alns]
    rows_fstsp.append({"NODE_ID": start_idx_for_alns, "NODE_TYPE": "truck_pos", "ORIG_X": float(nd_start.get('x', 0)),
                       "ORIG_Y": float(nd_start.get('y', 0)), "EFFECTIVE_DUE": 1e9})

    nd_depot = data_next.nodes[data_next.central_idx]
    rows_fstsp.append({"NODE_ID": data_next.central_idx, "NODE_TYPE": "central", "ORIG_X": float(nd_depot.get('x', 0)),
                       "ORIG_Y": float(nd_depot.get('y', 0)), "EFFECTIVE_DUE": 1e9})

    for c in unserved_customers:
        nd_c = data_next.nodes[c]
        rows_fstsp.append({"NODE_ID": int(c), "NODE_TYPE": "customer", "ORIG_X": float(nd_c.get('x', 0)),
                           "ORIG_Y": float(nd_c.get('y', 0)),
                           "EFFECTIVE_DUE": float(nd_c.get('effective_due', nd_c.get('due_time', 0.0)))})

    df_sub_fstsp = pd.DataFrame(rows_fstsp)

    # 求解残局
    res = fstsp_solver.solve_fstsp_return_from_df(
        df_sub_fstsp,
        E_roundtrip_km=float(ab_cfg.get("E_roundtrip_km", 10.0)),
        truck_speed_kmh=float(ab_cfg.get("truck_speed_kmh", 30.0)),
        truck_road_factor=float(ab_cfg.get("truck_road_factor", sim.TRUCK_ROAD_FACTOR)),
        drone_speed_kmh=float(ab_cfg.get("drone_speed_kmh", 60.0)),
        alpha=float(ab_cfg.get("alpha_drone", 0.3)),
        lambda_late=float(ab_cfg.get("lambda_late", 50.0)),
        time_limit=float(ab_cfg.get("grb_time_limit", 600.0)),
        mip_gap=float(ab_cfg.get("grb_mip_gap", 0.0)),
        unit_per_km=float(ab_cfg.get("unit_per_km", 5.0)),
        start_node=int(start_idx_for_alns),
        start_time_h=float(decision_time),
        verbose=int(ab_cfg.get("grb_verbose", 0))
    )

    # 合并前缀与后缀
    if int(res.get("sol_count", 0)) > 0:
        suffix_route = res["route"]
        if prefix_route and suffix_route and prefix_route[-1] == suffix_route[0]:
            full_route_next = prefix_route + suffix_route[1:]
        else:
            full_route_next = prefix_route + suffix_route

        full_triplets_next = _uniq_triplets(prefix_triplets + res.get("fstsp_triplets", []))
    else:
        full_route_next = full_route_cur
        full_triplets_next = full_triplets_cur

    # 全局最终评估
    full_eval = fstsp_evaluator.evaluate_fstsp_system(
        data_next, full_route_next, full_triplets_next,
        truck_speed_units=sim.TRUCK_SPEED_UNITS, drone_speed_units=sim.DRONE_SPEED_UNITS,
        truck_road_factor=sim.TRUCK_ROAD_FACTOR,
        alpha_drone=0.3, lambda_late=float(ab_cfg.get("lambda_late", 50.0))
    )

    stat_record = ut._pack_scene_record(
        scene_idx, decision_time, full_eval,
        num_req=len(req_override), num_acc=num_acc, num_rej=len(req_override) - num_acc,
        alpha_drone=0.3, lambda_late=50.0, solver_time=res.get("runtime_sec", 0.0)
    )

    return {
        'break': False, 'skip': False,
        'data_next': data_next,
        'full_route_next': full_route_next,
        'full_triplets_next': full_triplets_next,
        'full_arrival_next': full_eval.get('arrival_times', {}),
        'full_finish_next': full_eval.get('finish_times', {}),
        'full_depart_next': full_eval.get('depart_times', full_eval.get('depart', {})),
        'stat_record': stat_record,
        'decision_log_rows': [],
        'full_eval': full_eval,
        'viz_pack': {
            'data': data_next,
            'route': full_route_next,
            'triplets': full_triplets_next,
            'decisions': decisions,
            'virtual_pos': virtual_pos,
            'prefix_route': prefix_route
        }
    }