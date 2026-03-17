# -*- coding: utf-8 -*-
"""
经典伴飞模式 (FSTSP) 动态重规划评估脚本 (全景可视化 + 全局成本追踪版)
- 彻底补全了路径箭头、已走灰色历史、位置偏移痕迹。
- 引入全局前缀累加逻辑，将 Gurobi 的 Suffix Cost 转化为可直接对比的 Global Cost。
- 输出完美的动态汇总表，完全对齐 ALNS 评估口径。
"""

import pandas as pd
import numpy as np
import math
import gurobipy as gp
from gurobipy import GRB
import os
import matplotlib.pyplot as plt


def euclid(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def load_events(events_path):
    """加载 events.csv 并按时间分组"""
    if not os.path.exists(events_path):
        return {}
    df_ev = pd.read_csv(events_path)
    groups = {}
    for _, row in df_ev.iterrows():
        t = float(row['EVENT_TIME'])
        t_key = round(t, 6)
        if t_key not in groups:
            groups[t_key] = []
        groups[t_key].append({
            'NODE_ID': int(row['NODE_ID']),
            'NEW_X': float(row['NEW_X']),
            'NEW_Y': float(row['NEW_Y'])
        })
    return groups


def plot_fstsp_scene_advanced(scene_name, cost, depot_pos, current_truck_pos,
                              nodes_dict, unserved_ids, served_ids, moved_info,
                              truck_coords_seq, drone_triplets_coords):
    """高级动态可视化引擎：绘制带有方向箭头、灰色历史与扰动痕迹的全景路由"""
    plt.figure(figsize=(10, 8), dpi=120)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 1. 终点仓库 (黄方块)
    plt.scatter(depot_pos[0], depot_pos[1], c='yellow', marker='s', s=150, edgecolors='black', zorder=6,
                label='中心仓库 (Depot)')

    # 2. 卡车当前位置 (红星星，仅在动态场景显示)
    if current_truck_pos != depot_pos:
        plt.scatter(current_truck_pos[0], current_truck_pos[1], c='red', marker='*', s=200, edgecolors='black',
                    zorder=7, label='卡车当前位置')

    # 3. 绘制已服务的节点 (灰色)
    if served_ids:
        for nid in served_ids:
            cx, cy = nodes_dict[nid]['X'], nodes_dict[nid]['Y']
            plt.scatter(cx, cy, c='gray', s=40, alpha=0.5, zorder=4)
            plt.text(cx + 0.6, cy + 0.6, str(nid), fontsize=9, color='darkgray')

    # 4. 未服务客户点 (蓝色)
    for nid in unserved_ids:
        cx, cy = nodes_dict[nid]['X'], nodes_dict[nid]['Y']
        plt.scatter(cx, cy, c='blue', s=40, zorder=5)
        plt.text(cx + 0.6, cy + 0.6, str(nid), fontsize=9, color='dimgray')

    # 5. 绘制位置偏移痕迹 (黑色原位 -> 虚线箭头 -> 蓝色新位)
    if moved_info:
        for nid, (ox, oy) in moved_info.items():
            nx, ny = nodes_dict[nid]['X'], nodes_dict[nid]['Y']
            plt.scatter(ox, oy, c='black', s=30, alpha=0.6, zorder=5)
            if (ox, oy) != (nx, ny):
                plt.annotate("", xy=(nx, ny), xytext=(ox, oy),
                             arrowprops=dict(arrowstyle="->", color="gray", linestyle="--", lw=1.5), zorder=4)
            plt.text(ox, oy, f"old_{nid}", fontsize=8, color='black')

    # 绘制带箭头的线段辅助函数
    def draw_arrow(p1, p2, color, ls, lw):
        if p1 != p2:
            # mutation_scale 控制箭头头部大小
            plt.annotate("", xy=p2, xytext=p1,
                         arrowprops=dict(arrowstyle="-|>", color=color, linestyle=ls, lw=lw, mutation_scale=15),
                         zorder=3)

    # 6. 卡车全景路线 (红色实线箭头)
    for i in range(len(truck_coords_seq) - 1):
        p1 = truck_coords_seq[i]
        p2 = truck_coords_seq[i + 1]
        draw_arrow(p1, p2, 'red', '-', 2.5)
        if i == 0:
            plt.plot([], [], c='red', lw=2.5, label='卡车路径')

    # 7. 无人机全景路线 (浅蓝色虚线箭头)
    for idx, (p_i, p_j, p_k) in enumerate(drone_triplets_coords):
        draw_arrow(p_i, p_j, 'skyblue', '--', 2.0)
        draw_arrow(p_j, p_k, 'skyblue', '--', 2.0)
        if idx == 0:
            plt.plot([], [], c='skyblue', ls='--', lw=2.0, label='无人机路径')

    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05), fontsize=10)
    plt.title(f"{scene_name} FSTSP 全景路由图 - 全局 Cost: {cost:.3f}", fontsize=14)
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def solve_fstsp_subproblem(
        nodes_dict, unserved_ids, start_pos, start_time, depot_pos,
        E_roundtrip_km=10.0, truck_speed_kmh=30.0, truck_road_factor=1.5,
        drone_speed_kmh=60.0, alpha=0.3, lambda_late=50.0,
        time_limit=1800.0, mip_gap=0.0, unit_per_km=5.0, verbose=0
):
    c = len(unserved_ids)
    nodeNum = c + 2

    corX = [0.0] * nodeNum
    corY = [0.0] * nodeNum
    due_time = [999999.0] * nodeNum
    idx2id = {}

    idx2id[0] = -1  # 虚拟起点 (当前卡车位置)
    corX[0], corY[0] = start_pos[0], start_pos[1]

    for i, nid in enumerate(unserved_ids):
        idx = i + 1
        idx2id[idx] = nid
        corX[idx] = nodes_dict[nid]['X']
        corY[idx] = nodes_dict[nid]['Y']
        due_time[idx] = nodes_dict[nid]['DUE']

    idx2id[c + 1] = 0  # 终点仓库
    corX[c + 1], corY[c + 1] = depot_pos[0], depot_pos[1]

    N_out_lst = list(range(0, c + 1))
    N_in_lst = list(range(1, c + 2))
    C_lst = list(range(1, c + 1))

    distMatrix = np.zeros((nodeNum, nodeNum))
    costMatrix = np.zeros((nodeNum, nodeNum))

    for i in range(nodeNum):
        for j in range(nodeNum):
            if i != j:
                d_units = euclid(corX[i], corY[i], corX[j], corY[j])
                distMatrix[i][j] = d_units / unit_per_km
                costMatrix[i][j] = (distMatrix[i][j] * truck_road_factor) / truck_speed_kmh

    UAV_factor = (1.0 / drone_speed_kmh) / (truck_road_factor / truck_speed_kmh)

    P_lst = []
    for i in N_out_lst:
        for j in C_lst:
            if i != j:
                for k in N_in_lst:
                    if k != j and k != i:
                        fly_time_ij = distMatrix[i][j] / drone_speed_kmh
                        fly_time_jk = distMatrix[j][k] / drone_speed_kmh
                        if fly_time_ij + fly_time_jk <= (E_roundtrip_km / drone_speed_kmh) + 1e-6:
                            P_lst.append((i, j, k))

    m = gp.Model('FSTSP_Subproblem')
    m.setParam('OutputFlag', verbose)
    m.setParam('TimeLimit', time_limit)
    m.setParam('MIPGap', mip_gap)

    M = 100000.0
    X = m.addVars(nodeNum, nodeNum, vtype=GRB.BINARY, name="X")
    Y = m.addVars(nodeNum, nodeNum, nodeNum, vtype=GRB.BINARY, name="Y")
    U = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=1.0, ub=c + 2, name="U")
    T = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=0.0, name="T")
    TUAV = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=0.0, name="TUAV")
    F = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=0.0, name="F")
    L = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=0.0, name="Late")

    for j in C_lst:
        m.addConstr(gp.quicksum(X[i, j] for i in N_out_lst if i != j) +
                    gp.quicksum(Y[i, j, k] for i in N_out_lst for k in N_in_lst if (i, j, k) in P_lst and i != j) == 1)

    m.addConstr(gp.quicksum(X[0, j] for j in N_in_lst) == 1)
    m.addConstr(gp.quicksum(X[i, c + 1] for i in N_out_lst) == 1)

    for j in C_lst:
        m.addConstr(
            gp.quicksum(X[i, j] for i in N_out_lst if i != j) == gp.quicksum(X[j, k] for k in N_in_lst if k != j))

    for i in C_lst:
        for j in N_in_lst:
            if i != j:
                m.addConstr(U[i] - U[j] + (c + 2) * X[i, j] <= c + 1)

    for i in N_out_lst:
        m.addConstr(gp.quicksum(Y[i, j, k] for j in C_lst for k in N_in_lst if (i, j, k) in P_lst and j != i) <= 1)
    for k in N_in_lst:
        m.addConstr(gp.quicksum(Y[i, j, k] for i in N_out_lst for j in C_lst if (i, j, k) in P_lst and i != k) <= 1)

    for i in C_lst:
        for j in C_lst:
            if j != i:
                for k in N_in_lst:
                    if (i, j, k) in P_lst:
                        m.addConstr(gp.quicksum(X[h, i] for h in N_out_lst if h != i) + gp.quicksum(
                            X[l, k] for l in C_lst if l != k) >= 2 * Y[i, j, k])

    for j in C_lst:
        for k in N_in_lst:
            if (0, j, k) in P_lst:
                m.addConstr(gp.quicksum(X[h, k] for h in N_out_lst if h != k) >= Y[0, j, k])

    for i in C_lst:
        m.addConstr(
            T[i] - TUAV[i] <= M * (1 - gp.quicksum(Y[i, j, k] for j in C_lst for k in N_in_lst if (i, j, k) in P_lst)))
        m.addConstr(
            TUAV[i] - T[i] <= M * (1 - gp.quicksum(Y[i, j, k] for j in C_lst for k in N_in_lst if (i, j, k) in P_lst)))
    for k in N_in_lst:
        m.addConstr(
            T[k] - TUAV[k] <= M * (1 - gp.quicksum(Y[i, j, k] for i in N_out_lst for j in C_lst if (i, j, k) in P_lst)))
        m.addConstr(
            TUAV[k] - T[k] <= M * (1 - gp.quicksum(Y[i, j, k] for i in N_out_lst for j in C_lst if (i, j, k) in P_lst)))

    for h in N_out_lst:
        for k in N_in_lst:
            if h != k:
                m.addConstr(T[h] - T[k] + costMatrix[h][k] <= M * (1 - X[h, k]))

    for j in C_lst:
        for i in N_out_lst:
            if i != j:
                m.addConstr(TUAV[i] - TUAV[j] + UAV_factor * costMatrix[i][j] <= M * (
                            1 - gp.quicksum(Y[i, j, k] for k in N_in_lst if (i, j, k) in P_lst)))
        for k in N_in_lst:
            if k != j:
                m.addConstr(TUAV[j] - TUAV[k] + UAV_factor * costMatrix[j][k] <= M * (
                            1 - gp.quicksum(Y[i, j, k] for i in N_out_lst if (i, j, k) in P_lst)))

    for k in N_in_lst:
        for j in C_lst:
            if j != k:
                for i in N_out_lst:
                    if (i, j, k) in P_lst:
                        m.addConstr(TUAV[k] - TUAV[j] + UAV_factor * costMatrix[i][j] <= (
                                    E_roundtrip_km / drone_speed_kmh) + M * (1 - Y[i, j, k]))

    m.addConstr(T[0] == start_time)
    m.addConstr(TUAV[0] == start_time)

    for j in C_lst:
        m.addConstr(F[j] >= T[j] - M * (1 - gp.quicksum(X[i, j] for i in N_out_lst if i != j)))
        for i in N_out_lst:
            if i != j:
                for k in N_in_lst:
                    if (i, j, k) in P_lst:
                        m.addConstr(F[j] >= TUAV[i] + UAV_factor * costMatrix[i][j] - M * (1 - Y[i, j, k]))
        m.addConstr(L[j] >= F[j] - due_time[j])

    truck_dist_units = gp.quicksum(
        distMatrix[i][j] * unit_per_km * truck_road_factor * X[i, j] for i in N_out_lst for j in N_in_lst if i != j)
    drone_dist_units = gp.quicksum(
        (distMatrix[i][j] * unit_per_km + distMatrix[j][k] * unit_per_km) * Y[i, j, k] for i in N_out_lst for j in C_lst
        for k in N_in_lst if (i, j, k) in P_lst)
    late_sum_h = gp.quicksum(L[j] for j in C_lst)

    m.setObjective(truck_dist_units + alpha * drone_dist_units + lambda_late * late_sum_h, GRB.MINIMIZE)
    m.optimize()

    if m.SolCount <= 0:
        return None

    truck_route_idx = [0]
    curr = 0
    while curr != c + 1:
        for j in N_in_lst:
            if X[curr, j].X > 0.5:
                truck_route_idx.append(j)
                curr = j
                break

    truck_route_ids = [idx2id[n] for n in truck_route_idx]

    drone_triplets = []
    for i in N_out_lst:
        for j in C_lst:
            for k in N_in_lst:
                if (i, j, k) in P_lst and Y[i, j, k].X > 0.5:
                    drone_triplets.append((idx2id[i], idx2id[j], idx2id[k]))

    node_arrival = {}
    for idx in range(1, c + 1):
        nid = idx2id[idx]
        node_arrival[nid] = F[idx].X

    res_dict = {
        'cost': m.ObjVal,
        'truck_dist': truck_dist_units.getValue(),
        'drone_dist': drone_dist_units.getValue(),
        'system_time': T[c + 1].X,
        'total_late': late_sum_h.getValue(),
        'node_arrival': node_arrival,
        'truck_route_ids': truck_route_ids,
        'drone_triplets': drone_triplets,
        'runtime': getattr(m, 'Runtime', 0.0)
    }

    return res_dict


def run_dynamic_fstsp(csv_path, events_path, decision_times=[1.0, 2.0], time_limit=1800.0):
    """动态滚动主框架 (搭载全局追踪能力)"""
    print(f"\n================= 经典伴飞 (FSTSP) 动态环境 ==================")

    # ==== 仿真超参 (与 ALNS 主文件严格对齐) ====
    UNIT_PER_KM = 5.0
    TRUCK_ROAD_FACTOR = 1.5
    ALPHA = 0.3
    LAMBDA_LATE = 50.0

    df = pd.read_csv(csv_path)
    df_cust = df[df['NODE_TYPE'].str.lower() == 'customer'].reset_index(drop=True)
    depot_row = df[df['NODE_TYPE'].str.lower() == 'central'].iloc[0]
    depot_pos = (float(depot_row['ORIG_X']), float(depot_row['ORIG_Y']))

    nodes_dict = {}
    for i in range(len(df_cust)):
        nid = int(df_cust.loc[i, 'NODE_ID'])
        nodes_dict[nid] = {
            'X': float(df_cust.loc[i, 'ORIG_X']),
            'Y': float(df_cust.loc[i, 'ORIG_Y']),
            'DUE': float(df_cust.loc[i, 'EFFECTIVE_DUE']) if 'EFFECTIVE_DUE' in df_cust.columns else float(
                df_cust.loc[i, 'DUE_TIME'])
        }

    events_grouped = load_events(events_path)
    unserved_ids = list(nodes_dict.keys())
    all_served_ids = []
    summary_data = []

    # --- 全局前缀追踪变量 ---
    prefix_truck_dist = 0.0
    prefix_drone_dist = 0.0
    prefix_late = 0.0
    last_truck_node = 0  # 初始在仓库
    prefix_truck_coords = [depot_pos]
    prefix_drone_triplets_coords = []

    prev_truck_route_ids = []
    prev_drone_triplets = []

    def get_coord(nid):
        if nid == 0: return depot_pos
        if nid == -1: return get_coord(last_truck_node)
        return (nodes_dict[nid]['X'], nodes_dict[nid]['Y'])

    current_truck_pos = depot_pos

    # [场景 0: 静态规划]
    print(f"\n>>> Scene 0: t=0.0h 静态初始求解...")
    res = solve_fstsp_subproblem(
        nodes_dict, unserved_ids, current_truck_pos, 0.0, depot_pos,
        time_limit=time_limit, verbose=0
    )

    if res is None:
        print("Scene 0 无解！")
        return

    # 初始化追踪变量
    prev_truck_route_ids = res['truck_route_ids']
    prev_drone_triplets = res['drone_triplets']
    planned_arrival = res['node_arrival']

    # 记录 Scene 0 数据 (初始没有 Prefix)
    global_cost = res['cost']
    print(
        f"    [Scene 0 结果] Cost: {global_cost:.3f} | System Time: {res['system_time']:.3f}h | Total Late: {res['total_late']:.3f}h")

    summary_data.append({
        'Scene': 0, 't_dec': 0.0, 'Req': 0,
        'Cost': global_cost, 'TruckDist': res['truck_dist'], 'DroneDist': res['drone_dist'],
        'TotalLate': res['total_late'], 'SysTime': res['system_time'], 'Runtime': res['runtime']
    })

    # 画图: Scene 0
    full_truck_coords = [get_coord(n) for n in prev_truck_route_ids]
    full_drone_coords = [(get_coord(i), get_coord(j), get_coord(k)) for i, j, k in prev_drone_triplets]
    plot_fstsp_scene_advanced("Scene 0", global_cost, depot_pos, current_truck_pos,
                              nodes_dict, unserved_ids, all_served_ids, {},
                              full_truck_coords, full_drone_coords)

    # 动态滚动
    for scene_idx, t_dec in enumerate(decision_times, start=1):
        print(f"\n>>> Scene {scene_idx}: 决策时刻 t={t_dec:.2f}h")

        # 1. 甄别并在排序后处理本轮已服务客户
        served_this_round = []
        for nid in list(unserved_ids):
            if planned_arrival.get(nid, 999) <= t_dec:
                served_this_round.append(nid)
                unserved_ids.remove(nid)
                all_served_ids.append(nid)

        served_this_round.sort(key=lambda x: planned_arrival[x])

        # 2. 结算 Prefix 指标 (赶在应用新坐标之前，用真实走过的历史坐标计算)
        for nid in served_this_round:
            arr_time = planned_arrival[nid]
            due = nodes_dict[nid]['DUE']
            if arr_time > due:
                prefix_late += (arr_time - due)

            if nid in prev_truck_route_ids:
                p1 = get_coord(last_truck_node)
                p2 = get_coord(nid)
                d_units = euclid(p1[0], p1[1], p2[0], p2[1])
                prefix_truck_dist += (d_units / UNIT_PER_KM) * TRUCK_ROAD_FACTOR
                prefix_truck_coords.append(p2)
                last_truck_node = nid
            else:
                triplet = next((t for t in prev_drone_triplets if t[1] == nid), None)
                if triplet:
                    i, j, k = triplet
                    pi, pj, pk = get_coord(i), get_coord(j), get_coord(k)
                    d1 = euclid(pi[0], pi[1], pj[0], pj[1])
                    d2 = euclid(pj[0], pj[1], pk[0], pk[1])
                    prefix_drone_dist += (d1 + d2) / UNIT_PER_KM
                    prefix_drone_triplets_coords.append((pi, pj, pk))

        # 更新卡车物理起点
        current_truck_pos = get_coord(last_truck_node)

        # 3. 注入位置变更事件
        evs = events_grouped.get(round(t_dec, 6), [])
        req_count = 0
        moved_info_this_round = {}

        for ev in evs:
            nid = ev['NODE_ID']
            if nid in unserved_ids:
                old_x, old_y = nodes_dict[nid]['X'], nodes_dict[nid]['Y']
                moved_info_this_round[nid] = (old_x, old_y)

                nodes_dict[nid]['X'] = ev['NEW_X']
                nodes_dict[nid]['Y'] = ev['NEW_Y']
                req_count += 1

        if req_count == 0:
            print(f"    [跳过] t={t_dec}h 时无有效扰动，跳过重规划。")
            continue

        print(f"    已接受 {req_count} 个客户的在线位置变更。开始 Gurobi 重规划残局...")

        # 4. 残局重规划
        res_sub = solve_fstsp_subproblem(
            nodes_dict, unserved_ids, current_truck_pos, t_dec, depot_pos,
            time_limit=time_limit, verbose=0
        )

        if res_sub is None:
            print("    [警告] 残局无解，FSTSP 陷入时空瘫痪！")
            break

        # 5. 计算 Global Metrics
        global_truck_dist = prefix_truck_dist + res_sub['truck_dist']
        global_drone_dist = prefix_drone_dist + res_sub['drone_dist']
        global_late = prefix_late + res_sub['total_late']
        global_cost = global_truck_dist + ALPHA * global_drone_dist + LAMBDA_LATE * global_late
        global_sys_time = res_sub['system_time']  # Suffix 里的 T[c+1] 已经是全局时间

        print(
            f"    [全局结果] Global Cost: {global_cost:.3f} | Global SysTime: {global_sys_time:.3f}h | Global Late: {global_late:.3f}h")

        summary_data.append({
            'Scene': scene_idx, 't_dec': t_dec, 'Req': req_count,
            'Cost': global_cost, 'TruckDist': global_truck_dist, 'DroneDist': global_drone_dist,
            'TotalLate': global_late, 'SysTime': global_sys_time, 'Runtime': res_sub['runtime']
        })

        # 6. 画图: 提取完整轨迹
        # 将 Suffix 中的 '-1' 映射为 'current_truck_pos'
        suffix_truck_coords = [current_truck_pos if n == -1 else get_coord(n) for n in res_sub['truck_route_ids']]
        full_truck_coords = prefix_truck_coords + suffix_truck_coords[1:]  # 避免首尾重复连接

        suffix_drone_coords = []
        for i, j, k in res_sub['drone_triplets']:
            pi = current_truck_pos if i == -1 else get_coord(i)
            pj = get_coord(j)
            pk = current_truck_pos if k == -1 else get_coord(k)
            suffix_drone_coords.append((pi, pj, pk))
        full_drone_coords = prefix_drone_triplets_coords + suffix_drone_coords

        plot_fstsp_scene_advanced(f"Scene {scene_idx}", global_cost, depot_pos, current_truck_pos,
                                  nodes_dict, unserved_ids, all_served_ids, moved_info_this_round,
                                  full_truck_coords, full_drone_coords)

        # 7. 更新追踪变量以备下一轮
        # 巧妙处理: 将残局的虚拟起点 '-1' 转化为真实的 last_truck_node ID，保证下一轮查询不出错
        prev_truck_route_ids = [last_truck_node if x == -1 else x for x in res_sub['truck_route_ids']]

        prev_drone_triplets = []
        for i, j, k in res_sub['drone_triplets']:
            i_clean = last_truck_node if i == -1 else i
            k_clean = last_truck_node if k == -1 else k
            prev_drone_triplets.append((i_clean, j, k_clean))

        planned_arrival.update(res_sub['node_arrival'])

    # --- 输出格式化汇总表 ---
    print(f"\n================= 经典伴飞 (FSTSP) 动态测试结束 ==================")
    print("\n===== 动态位置变更场景汇总 (FSTSP) =====")
    df_summary = pd.DataFrame(summary_data)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))


if __name__ == "__main__":
    # 请务必替换为你生成的 N=25 的数据路径
    test_nodes = r"D:\代码\ALNS+DL\exp\datasets\25_data\2023\nodes_25_seed2023_20260129_164341_promise.csv"
    test_events = r"D:\代码\ALNS+DL\exp\datasets\25_data\2023\events_25_seed2023_20260129_164341.csv"

    run_dynamic_fstsp(test_nodes, test_events, decision_times=[1.0, 2.0], time_limit=1800.0)