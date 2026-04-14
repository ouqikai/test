# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import time


def euclid(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)

def _sanitize_fstsp_triplets(route_ids, raw_triplets):
    """
    中文注释：
    对 Gurobi 解出来的原始 triplets 做一次后处理清洗，防止出现：
    1) 起飞点/回收点根本不在卡车 route 上；
    2) 起飞顺序晚于回收顺序；
    3) 同一个客户既在 truck route 中，又在 drone triplet 中；
    4) 同一个客户被多个 triplet 重复服务。
    """
    route_ids = [int(x) for x in (route_ids or [])]

    # 记录每个节点在 route 中出现的位置（central 可能出现两次）
    pos_map = {}
    for pos, nid in enumerate(route_ids):
        nid = int(nid)
        pos_map.setdefault(nid, []).append(pos)

    # FSTSP 的中间节点都是卡车访问客户；若某客户已经在卡车 route 中，就不能再作为 drone 客户
    truck_customer_set = set(int(x) for x in route_ids[1:-1])

    clean_triplets = []
    drone_assign = {}
    seen_triplets = set()
    seen_customers = set()

    for tri in raw_triplets or []:
        try:
            launch_id, cust_id, rendez_id = int(tri[0]), int(tri[1]), int(tri[2])
        except Exception:
            continue

        # 1) 同一客户不允许重复服务
        if cust_id in seen_customers:
            continue

        # 2) 客户已经在卡车 route 中，则该 triplet 非法
        if cust_id in truck_customer_set:
            continue

        # 3) 起飞点、回收点必须都在 route 中
        if launch_id not in pos_map or rendez_id not in pos_map:
            continue

        # 4) route 顺序上必须存在“起飞在前、回收在后”
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
        clean_triplets.append(tt)
        drone_assign.setdefault(launch_id, []).append(cust_id)

    return clean_triplets, drone_assign

def solve_fstsp_return_from_df(
        df: pd.DataFrame,
        E_roundtrip_km: float = 10.0,
        truck_speed_kmh: float = 30.0,
        truck_road_factor: float = 1.5,
        drone_speed_kmh: float = 60.0,
        alpha: float = 0.3,
        lambda_late: float = 50.0,
        time_limit: float = 1800.0,
        mip_gap: float = 0.0,
        unit_per_km: float = 5.0,
        start_node: int = None,
        start_time_h: float = 0.0,
        verbose: int = 0
):
    """
    纯粹的 FSTSP (伴飞模式) 求解器。
    输入：DataFrame (包含 truck_pos, central, customer)
    输出：字典格式，包含 route 和 drone_assign，完全兼容你的 ALNS 框架动态重规划需求。
    """
    t_start = time.time()

    # 解析 df，提取关键节点
    start_row = df[df['NODE_TYPE'] == 'truck_pos'].iloc[0]
    depot_row = df[df['NODE_TYPE'] == 'central'].iloc[0]
    df_cust = df[df['NODE_TYPE'] == 'customer']

    c = len(df_cust)
    nodeNum = c + 2

    corX = [0.0] * nodeNum
    corY = [0.0] * nodeNum
    due_time = [999999.0] * nodeNum
    idx2id = {}

    # 0 索引：虚拟起点
    idx2id[0] = int(start_row['NODE_ID'])
    corX[0], corY[0] = float(start_row['ORIG_X']), float(start_row['ORIG_Y'])

    # 1 到 c：客户点
    for i, (_, row) in enumerate(df_cust.iterrows()):
        idx = i + 1
        idx2id[idx] = int(row['NODE_ID'])
        corX[idx] = float(row['ORIG_X'])
        corY[idx] = float(row['ORIG_Y'])
        due_time[idx] = float(row['EFFECTIVE_DUE'])

    # c + 1：终点 (Depot)
    idx2id[c + 1] = int(depot_row['NODE_ID'])
    corX[c + 1], corY[c + 1] = float(depot_row['ORIG_X']), float(depot_row['ORIG_Y'])

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

    m = gp.Model('FSTSP_MILP')
    m.setParam('OutputFlag', verbose)
    m.setParam('TimeLimit', float(time_limit))
    m.setParam('MIPGap', float(mip_gap))

    M = 100000.0
    X = m.addVars(nodeNum, nodeNum, vtype=GRB.BINARY, name="X")
    Y = m.addVars(nodeNum, nodeNum, nodeNum, vtype=GRB.BINARY, name="Y")
    U = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=1.0, ub=c + 2, name="U")
    T = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=0.0, name="T")
    TUAV = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=0.0, name="TUAV")
    F = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=0.0, name="F")
    L = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=0.0, name="Late")
    # ==================== 从这里开始替换 ====================
    # 0. 基础覆盖与起终点约束（上次替换时被误删，必须补回！）
    # 每个客户必须被服务一次（要么被卡车直接访问，要么被无人机服务）
    for j in C_lst:
        m.addConstr(gp.quicksum(X[i, j] for i in N_out_lst if i != j) +
                    gp.quicksum(Y[i, j, k] for i in N_out_lst for k in N_in_lst if (i, j, k) in P_lst) == 1)

    # 卡车必须从起点(0)出发，且只出发一次
    m.addConstr(gp.quicksum(X[0, j] for j in N_in_lst) == 1)
    # 卡车必须回到终点(c+1)，且只回一次
    m.addConstr(gp.quicksum(X[i, c + 1] for i in N_out_lst) == 1)

    # 卡车在中间节点的流平衡约束
    for j in C_lst:
        m.addConstr(
            gp.quicksum(X[i, j] for i in N_out_lst if i != j) == gp.quicksum(X[j, k] for k in N_in_lst if k != j))

    # 1. 锚定起点的 U 变量，防止顺序链条悬空
    m.addConstr(U[0] == 1)

    # 2. MTZ 约束 (消除卡车子环)
    for i in N_out_lst:
        for j in N_in_lst:
            if i != j:
                m.addConstr(U[i] - U[j] + (c + 2) * X[i, j] <= c + 1)

    # 3. 强制拓扑时序：起飞点 i 必须在 回收点 k 之前被卡车访问 (彻底修复时空倒流)
    for i in N_out_lst:
        for j in C_lst:
            for k in N_in_lst:
                if (i, j, k) in P_lst:
                    m.addConstr(U[k] - U[i] >= 1 - M * (1 - Y[i, j, k]), name=f"topo_{i}_{j}_{k}")

    # 4. 每个起飞/回收点最多只能派发/回收一次无人机
    for i in N_out_lst:
        m.addConstr(gp.quicksum(Y[i, j, k] for j in C_lst for k in N_in_lst if (i, j, k) in P_lst) <= 1)
    for k in N_in_lst:
        m.addConstr(gp.quicksum(Y[i, j, k] for i in N_out_lst for j in C_lst if (i, j, k) in P_lst) <= 1)

    # 5. X 和 Y 的流关联：彻底修复起点 0 作为起飞点时被禁止的 Bug
    for i in N_out_lst:
        for j in C_lst:
            for k in N_in_lst:
                if (i, j, k) in P_lst:
                    # i 作为起飞点，必然有卡车从 i 【驶出】（完美兼容起点 0）
                    m.addConstr(Y[i, j, k] <= gp.quicksum(X[i, h] for h in N_in_lst if h != i))
                    # k 作为回收点，必然有卡车【驶入】 k（完美兼容终点 c+1）
                    m.addConstr(Y[i, j, k] <= gp.quicksum(X[l, k] for l in N_out_lst if l != k))

    # 6. 卡车时间传播约束
    for h in N_out_lst:
        for k in N_in_lst:
            if h != k:
                m.addConstr(T[h] - T[k] + costMatrix[h][k] <= M * (1 - X[h, k]))

    # 7. 绑定无人机时间与卡车时间
    for i in N_out_lst:
        m.addConstr(TUAV[i] == T[i])
    # ==================== 替换到这里结束 ====================
    m.addConstr(T[0] == start_time_h)
    m.addConstr(TUAV[0] == start_time_h)

    for j in C_lst:
        m.addConstr(F[j] >= T[j] - M * (1 - gp.quicksum(X[i, j] for i in N_out_lst if i != j)))
        for i in N_out_lst:
            if i != j:
                for k in N_in_lst:
                    if (i, j, k) in P_lst:
                        m.addConstr(F[j] >= TUAV[i] + UAV_factor * costMatrix[i][j] - M * (1 - Y[i, j, k]))
        m.addConstr(L[j] >= F[j] - due_time[j])

    # 为了和你的 ALNS 框架完全对齐，我们在目标函数里也使用 unit (而非 km) 为口径
    truck_dist_units = gp.quicksum(
        distMatrix[i][j] * unit_per_km * truck_road_factor * X[i, j] for i in N_out_lst for j in N_in_lst if i != j)
    drone_dist_units = gp.quicksum(
        (distMatrix[i][j] * unit_per_km + distMatrix[j][k] * unit_per_km) * Y[i, j, k] for i in N_out_lst for j in C_lst
        for k in N_in_lst if (i, j, k) in P_lst)
    late_sum_h = gp.quicksum(L[j] for j in C_lst)

    m.setObjective(truck_dist_units + alpha * drone_dist_units + lambda_late * late_sum_h, GRB.MINIMIZE)
    m.optimize()

    runtime = time.time() - t_start

    if m.SolCount <= 0:
        return {"sol_count": 0, "runtime_sec": runtime}

    # 提取卡车路线
    truck_route_idx = [0]
    curr = 0
    while curr != c + 1:
        for j in N_in_lst:
            if X[curr, j].X > 0.5:
                truck_route_idx.append(j)
                curr = j
                break

    truck_route_ids = [idx2id[n] for n in truck_route_idx]

    # 先提取原始 triplets，再按卡车 route 做合法性清洗
    raw_triplets = []
    for i in N_out_lst:
        for j in C_lst:
            for k in N_in_lst:
                if (i, j, k) in P_lst and Y[i, j, k].X > 0.5:
                    launch_node_id = idx2id[i]
                    cust_node_id = idx2id[j]
                    rendezvous_id = idx2id[k]
                    raw_triplets.append((launch_node_id, cust_node_id, rendezvous_id))

    fstsp_triplets, drone_assign = _sanitize_fstsp_triplets(truck_route_ids, raw_triplets)

    return {
        "status": m.Status,
        "sol_count": m.SolCount,
        "route": truck_route_ids,
        "drone_assign": drone_assign,
        "fstsp_triplets": fstsp_triplets,  # <--- 新增这行输出
        "obj": m.ObjVal,
        "runtime_sec": runtime
    }