import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

def build_graph_batch(avg_scores, edge_mask, node_mask, confidence=None):
    """
    avg_scores: [B, N, N]
    edge_mask: [B, N, N]
    node_mask: [B, N] 
    """
    B, N, _ = avg_scores.shape
    graph_data_batch = []

    for b in range(B):
        A = avg_scores[b]
        E = edge_mask[b].bool()
        V = node_mask[b].bool()

        valid_idx = torch.nonzero(V, as_tuple=True)[0]   # [K]
        K = len(valid_idx)
        if confidence is not None:
            confidence = confidence[b, valid_idx]
        if K == 0:
            graph_data_batch.append({
                "edge_index": torch.empty(2, 0, dtype=torch.long, device=A.device),
                "edge_weight": torch.empty(0, device=A.device),
                "adj_weight": torch.zeros(0, 0, device=A.device),
                "graph": [],
                "mask": torch.zeros(0, 0, dtype=torch.bool, device=A.device),
                "node_ids": valid_idx,
            })
            continue

        A_sub = A[valid_idx][:, valid_idx]     # [K, K]
        E_sub = E[valid_idx][:, valid_idx]     # [K, K]

        adj_weight = A_sub * E_sub.float()

        dst, src = torch.nonzero(E_sub, as_tuple=True)
        edge_weight = adj_weight[dst, src]
        edge_index = torch.stack([dst, src], dim=0)

        graph = [[] for _ in range(K)]
        for v, u, w in zip(dst.tolist(), src.tolist(), edge_weight.tolist()):
            graph[u].append((v, w))

        graph_data_batch.append({
            "edge_index": edge_index,      # [2, E] in compressed index
            "edge_weight": edge_weight,    # [E]
            "adj_weight": adj_weight,      # [K, K]
            "graph": graph,                # compressed adjacency list
            "mask": E_sub,                 # [K, K]
            "node_ids": valid_idx,         # global idx mapping
            "confidence": confidence,      # [K]
        })

    return graph_data_batch

def tarjan_scc(graph):
    """
    graph: adjacency list, graph[u] = [(v, w), ...]
    return: list of SCCs, each SCC is a list of nodes
    """
    N = len(graph)
    # print(f'N: {N}')
    index = 0
    indices = [-1] * N
    lowlink = [0] * N
    stack = []
    on_stack = [False] * N
    scc_list = []

    def strongconnect(v):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True

        for (w, _) in graph[v]:
            if indices[w] == -1:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], indices[w])

        # If v is root of SCC
        if lowlink[v] == indices[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            scc_list.append(sorted(scc))  # sort for consistency

    for v in range(N):
        if indices[v] == -1:
            strongconnect(v)

    return scc_list

def compress_to_scc_dag(graph, scc_list):
    """
    graph: adjacency list
    scc_list: list of lists, SCCs
    return: dag_adj_list, scc_id_of_node
    """
    N = len(graph)
    scc_id = {}
    for i, comp in enumerate(scc_list):
        for u in comp:
            scc_id[u] = i

    K = len(scc_list)
    dag = [[] for _ in range(K)]
    dag_edges = set()

    for u in range(N):
        for v, _ in graph[u]:
            cu = scc_id[u]
            cv = scc_id[v]
            if cu != cv:
                if (cu, cv) not in dag_edges:
                    dag[cu].append(cv)
                    dag_edges.add((cu, cv))

    return dag, scc_id

def topo_sort_scc(dag):
    K = len(dag)
    indeg = [0] * K

    # compute indegree
    for u in range(K):
        for v in dag[u]:
            indeg[v] += 1

    # initial SCCs with no dependencies
    queue = [u for u in range(K) if indeg[u] == 0]

    levels = []

    # multi-level parallel schedule
    while queue:
        next_q = []
        levels.append(queue.copy())

        for u in queue:
            for v in dag[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    next_q.append(v)

        queue = next_q

    return levels

def safe_parallel_schedule_scc(graph_data):
    """
    graph_data from build_graph_batch()[b]
    returns: list of parallel groups (each group is a list of tokens)
    """
    graph = graph_data["graph"]          # adjacency list

    scc_list = tarjan_scc(graph)
    # print(f'scc_list: {scc_list}, length: {len(scc_list)}')

    dag, scc_id = compress_to_scc_dag(graph, scc_list)
    # print(f'dag: {dag}, length: {len(dag)}')

    dag_levels = topo_sort_scc(dag)
    # print(f'dag_levels: {dag_levels}, length: {len(dag_levels)}')

    # final_groups = []
    # for level in dag_levels:
    #     group = []
    #     for cid in level:
    #         group.extend(scc_list[cid])
    #     final_groups.append(sorted(group))

    selected_index = []
    selected_level = dag_levels[0]

    for cid in selected_level:
        scc = scc_list[cid]
        scc_confidence = graph_data["confidence"][scc]
        _, local_index = torch.topk(scc_confidence, k=1)
        select_index_scc_ = scc[local_index]
        select_index_scc = graph_data["node_ids"][select_index_scc_]
        # assert select_index_scc in selected_index, f'scc list is {scc_list}, selected level is {selected_level}, select index is {select_index_scc}, select index_scc is {select_index_scc_}, selected index is {selected_index}'
        selected_index.append(select_index_scc)

    # print(f'selected_index: {selected_index}, length: {len(selected_index)}')
    return selected_index
    
# def select_parallel_tokens_conflict_mis(
#     graph_data,
#     tau_dep: float = 0.0,
#     max_parallel: int | None = None,
# ):
#     A = graph_data["adj_weight"]
#     node_ids = graph_data["node_ids"]
#     confidence = graph_data["confidence"]

#     K = A.shape[0]
#     if K == 0:
#         return []

#     dep = torch.maximum(A, A.T)
#     conflict = dep > tau_dep
#     conflict.fill_diagonal_(False)

#     candidates = set(range(K))
#     parallel_comp = []

#     if max_parallel is None:
#         max_parallel = K

#     while candidates and len(parallel_comp) < max_parallel:
#         cand_list = list(candidates)
#         cand_scores = confidence[cand_list]
#         best_local_idx = torch.argmax(cand_scores).item()
#         best_node = cand_list[best_local_idx]

#         parallel_comp.append(best_node)

#         # neigh_bool = conflict[best_node] | conflict[:, best_node]
#         neigh_bool = conflict[best_node]
#         neigh_idx = torch.nonzero(neigh_bool, as_tuple=True)[0].tolist()

#         to_remove = set(neigh_idx)
#         to_remove.add(best_node)
#         candidates -= to_remove

#     parallel_global = [int(node_ids[i].item()) for i in parallel_comp]

#     return parallel_global

def select_parallel_tokens_conflict_mis(edge_mask, node_mask, confidence, max_parallel=None):
    K = node_mask.sum().item()
    if K == 0:
        return []

    conflict = edge_mask | edge_mask.T

    select_index = []

    if max_parallel is None:
        max_parallel = K

    while len(select_index) < max_parallel:
        best_node_idx = torch.argmax(confidence).item()
        select_index.append(best_node_idx)
        confidence[best_node_idx] = -np.inf

        neigh_bool = conflict[best_node_idx]
        neigh_idx = torch.nonzero(neigh_bool, as_tuple=True)[0].tolist()
        node_mask[neigh_idx] = False
        confidence[neigh_idx] = -np.inf

    return select_index

def mean_field(logits, avg_scores, beta=0.2, T=2):
    W_dir = avg_scores.transpose(1, 2)
    log_psi = logits
    q = F.softmax(logits, dim=-1)

    for _ in range(T):
        neighbor_msg = torch.matmul(W_dir, q)  # [B, L, V]
        log_q_new = log_psi + beta * neighbor_msg   # [B, L, V]
        q_new = F.softmax(log_q_new, dim=-1)
        q = q_new

    return log_q_new

def detect_attn_sinks(attn_scores, ratio=None, topk=None):
    # 1) 对每个被指向的 token j，算它被所有 i 看的平均 attention
    # barA[b, j] = mean_i A[b, i, j]
    # 在维度 1 上取 mean（query 维），保留 key 维 j
    B, L, _ = attn_scores.shape

    # 1) barA[b, j] = mean_i A[b, i, j]
    barA = attn_scores.mean(dim=1)      # [B, L]
    global_mask = torch.zeros_like(barA, dtype=torch.bool)

    if ratio is not None and ratio > 0:
        num_tokens = barA.shape[1]
        k_global = max(1, int(num_tokens * ratio))
        _, global_idx = torch.topk(barA, k=k_global, dim=-1)
        global_mask.scatter_(1, global_idx, True)

    if topk is not None and topk > 0:
        topk = min(topk, L)
        _, per_seq_top_idx = torch.topk(barA, k=topk, dim=-1)  # [B, topk]
        global_mask.scatter_(1, per_seq_top_idx, True)

    return global_mask