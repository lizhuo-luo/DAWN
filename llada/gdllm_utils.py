import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

import matplotlib.pyplot as plt
import seaborn as sns

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

def detect_attn_sinks(attn_scores, ratio=None, topk=None):
    B, L, _ = attn_scores.shape
    barA = attn_scores.mean(dim=1)  # [B, L]

    k1 = 0
    if ratio is not None and ratio > 0:
        k1 = max(1, int(L * ratio))
    k2 = topk if (topk is not None and topk > 0) else 0
    k = min(L, max(k1, k2))

    global_mask = torch.zeros((B, L), device=attn_scores.device, dtype=torch.bool)
    if k > 0:
        _, idx = torch.topk(barA, k=k, dim=-1)
        global_mask.scatter_(1, idx, True)
    return global_mask

def detect_attn_sinks_(attn_scores, threshold=0.05):
    """
    attn_scores: [B, S, S] softmax attention
    """
    B, S, _ = attn_scores.shape

    # column mean
    barA = attn_scores.mean(dim=1)          # [B, S]
    return barA > threshold

