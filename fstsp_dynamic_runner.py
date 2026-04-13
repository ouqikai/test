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
def _sanitize_triplets_for_route(data, route, triplets, forbidden_customers=None):
    """
    中文注释：
    按“当前 full route”过滤 triplets，避免 triplet 与 route 脱节。
    forbidden_customers:
        - 可传入“本轮决策前已经服务完成的客户集合”，用于禁止这些客户再次进入 suffix triplets。
    """
    route = [int(x) for x in (route or [])]
    forbidden_customers = set(int(x) for x in (forbidden_customers or []))

    pos_map = {}
    for pos, nid in enumerate(route):
        nid = int(nid)
        pos_map.setdefault(nid, []).append(pos)

    truck_customer_set = set()
    for nid in route:
        try:
            nid = int(nid)
            if 0 <= nid < len(data.nodes):
                if str(data.nodes[nid].get("node_type", "")).lower() == "customer":
                    truck_customer_set.add(nid)
        except Exception:
            continue

    clean = []
    seen_triplets = set()
    seen_customers = set()

    for tri in _uniq_triplets(triplets):
        try:
            launch_id, cust_id, rendez_id = int(tri[0]), int(tri[1]), int(tri[2])
        except Exception:
            continue

        # 已经在决策前完成服务的客户，不允许再出现在新后缀 triplet 中
        if cust_id in forbidden_customers:
            continue

        # 同一客户不允许重复服务
        if cust_id in seen_customers:
            continue

        # 若该客户已经在卡车 route 中，则不应再作为无人机客户
        if cust_id in truck_customer_set:
            continue

        # 起飞/回收节点必须都在 route 中
        if launch_id not in pos_map or rendez_id not in pos_map:
            continue

        # 必须存在一组位置满足：起飞在前、回收在后
        feasible_order = False
        for lp in pos_map[launch_id]:
            for rp in pos_map[rendez_id]:
                if lp < rp:
                    feasible_order = True
                    break
            if feasible_order:
                break
        if not feasible_order:
            continue

        tt = (launch_id, cust_id, rendez_id)
        if tt in seen_triplets:
            continue

        seen_triplets.add(tt)
        seen_customers.add(cust_id)
        clean.append(tt)

    return clean

def _future_triplets_for_viz(triplet_times, decision_time):
    """
    中文注释：
    只把“当前决策时刻之后仍未回收完成”的 triplet 返回给可视化，
    避免历史已完成 triplet 在图上继续显示，造成“又送了一次”的错觉。
    """
    out = []
    for tri, tt in (triplet_times or {}).items():
        try:
            recover_t = float(tt.get("recover_time", float("inf")))
        except Exception:
            continue
        if recover_t > float(decision_time) + 1e-9:
            try:
                out.append((int(tri[0]), int(tri[1]), int(tri[2])))
            except Exception:
                continue
    return _uniq_triplets(out)

def _eval_and_prune_triplets(data, route, triplets, ab_cfg, scene_idx):
    """
    中文注释：
    先静默评估一次，只保留 evaluator 真正执行成功的 triplets。
    这样可以把无效 triplet 在“当前场景内部”就删掉，
    避免它们先打印 warning，再拖到下一幕才清理。
    """
    triplets = _uniq_triplets(triplets)

    eval_res = fstsp_evaluator.evaluate_fstsp_system(
        data, route, triplets,
        truck_speed_units=sim.TRUCK_SPEED_UNITS,
        drone_speed_units=sim.DRONE_SPEED_UNITS,
        truck_road_factor=sim.TRUCK_ROAD_FACTOR,
        alpha_drone=float(ab_cfg.get("alpha_drone", 0.3)),
        lambda_late=float(ab_cfg.get("lambda_late", 50.0)),
        warn_missing=False,
        debug_missing=False
    )

    executed = set()
    for tri in eval_res.get("triplet_times", {}).keys():
        try:
            executed.add((int(tri[0]), int(tri[1]), int(tri[2])))
        except Exception:
            continue

    clean = []
    for tri in triplets:
        try:
            tt = (int(tri[0]), int(tri[1]), int(tri[2]))
        except Exception:
            continue
        if tt in executed:
            clean.append(tt)

    clean = _uniq_triplets(clean)

    if len(clean) != len(triplets):
        print(f"[FSTSP-PRUNE] scene={scene_idx} prune {len(triplets) - len(clean)} invalid triplets")
        eval_res = fstsp_evaluator.evaluate_fstsp_system(
            data, route, clean,
            truck_speed_units=sim.TRUCK_SPEED_UNITS,
            drone_speed_units=sim.DRONE_SPEED_UNITS,
            truck_road_factor=sim.TRUCK_ROAD_FACTOR,
            alpha_drone=float(ab_cfg.get("alpha_drone", 0.3)),
            lambda_late=float(ab_cfg.get("lambda_late", 50.0)),
            warn_missing=False,
            debug_missing=False
        )
    bad = [tt for tt in triplets if tuple(map(int, tt)) not in executed]
    if bad:
        print(f"[FSTSP-PRUNE-DETAIL] scene={scene_idx} bad={bad[:5]}")
    return clean, eval_res
def fstsp_quick_filter(data_prelim, full_route_cur, full_triplets_cur, decisions_raw, ab_cfg):
    """FSTSP 专属的快速过滤器（修复：兼容 8 元组解包，并规范化返回格式）"""
    res_work = fstsp_evaluator.evaluate_fstsp_system(
        data_prelim, full_route_cur, full_triplets_cur,
        truck_speed_units=sim.TRUCK_SPEED_UNITS, drone_speed_units=sim.DRONE_SPEED_UNITS,
        truck_road_factor=sim.TRUCK_ROAD_FACTOR,
        alpha_drone=float(ab_cfg.get("alpha_drone", 0.3)),
        lambda_late=float(ab_cfg.get("lambda_late", 50.0)), warn_missing=False, debug_missing=False
    )
    base_cost = res_work["cost"]
    base_late = res_work["total_late"]

    data_work = data_prelim
    decisions_filtered = []

    qf_cost_max = float(ab_cfg.get("qf_cost_max", 30.0))
    qf_late_max = float(ab_cfg.get("qf_late_max", 0.5))

    # 🚨 核心修复：兼容 decisions_raw 里的 8 元组或 5 元组
    for it in decisions_raw:
        if len(it) == 8:
            cid, nid, dec, reason, ox, oy, nx, ny = it
        elif len(it) == 5:
            cid, dec, nx, ny, reason = it
        else:
            continue

        if dec != "ACCEPT":
            # 外部要求返回 5 元组以方便做统计
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
            lambda_late=float(ab_cfg.get("lambda_late", 50.0)), warn_missing=False, debug_missing=False
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

    # 0. 先把当前 triplets 按 full_route_cur 做一次清洗，避免脏状态继续传下去
    full_triplets_cur = _sanitize_triplets_for_route(
        data_cur, full_route_cur, full_triplets_cur
    )

    full_triplets_cur, eval_cur = _eval_and_prune_triplets(
        data_cur, full_route_cur, full_triplets_cur, ab_cfg, scene_idx
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
        viz_triplets_hold = _future_triplets_for_viz(triplet_times, decision_time)
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
                'triplets': viz_triplets_hold,
                'decisions': decisions,
                'virtual_pos': virtual_pos,
                'prefix_route': prefix_route
            }
        }
    if active_triplets:
        full_triplets_hold, full_eval_hold = _eval_and_prune_triplets(
            data_next, full_route_cur, full_triplets_cur, ab_cfg, scene_idx
        )
        viz_triplets_hold = _future_triplets_for_viz(
            full_eval_hold.get("triplet_times", {}),
            decision_time
        )
        return {
            'break': False,
            'skip': True,
            'data_next': data_next,
            'full_route_next': full_route_cur,
            'full_triplets_next': full_triplets_hold,
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
                'triplets': viz_triplets_hold,
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

        # 先清洗“历史前缀 triplets”
        prefix_triplets = _sanitize_triplets_for_route(
            data_cur, full_route_cur, prefix_triplets
        )

        # 再清洗“新求解出来的后缀 triplets”
        # 关键：本轮决策前已经完成服务的客户，禁止再次进入 suffix triplets

        suffix_triplets = _sanitize_triplets_for_route(
            data_next, full_route_next, res.get("fstsp_triplets", []),
            forbidden_customers=served_customers
        )
        suffix_triplets, _ = _eval_and_prune_triplets(
            data_next, full_route_next, suffix_triplets, ab_cfg, scene_idx
        )
        full_triplets_next = _uniq_triplets(prefix_triplets + suffix_triplets)
    else:
        full_route_next = full_route_cur
        full_triplets_next = full_triplets_cur

    full_triplets_next, full_eval = _eval_and_prune_triplets(
        data_next, full_route_next, full_triplets_next, ab_cfg, scene_idx
    )
    # 仅用于画图：只展示当前时刻之后仍未完成的 triplets
    viz_triplets_next = _future_triplets_for_viz(
        full_eval.get("triplet_times", {}),
        decision_time
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
            'triplets': viz_triplets_next,
            'decisions': decisions,
            'virtual_pos': virtual_pos,
            'prefix_route': prefix_route
        }
    }