import torch

def get_threshold_mask(conf, unlabel_mask, strategy='dynamic_mean', threshold_value=0.7):
    """
    置信度筛选逻辑。论文 IceBerg Eq (5) + Algorithm 1: τ' = (1/|V_U|) Σ_j max(f_θ(v_j))。
    :param strategy: 'dynamic_mean' (论文动态阈值), 'global_mean', 'fixed'
    """
    conf_unlabeled = conf[unlabel_mask]
    if strategy == 'dynamic_mean':
        # 论文 Eq (5): 动态阈值 = 未标注节点上的置信度均值
        threshold = conf_unlabeled.mean().item() if conf_unlabeled.numel() > 0 else 0.0
    elif strategy == 'global_mean':
        # 均值策略变体（研究用）
        threshold = conf_unlabeled.mean().item()
    elif strategy == 'fixed':
        # 固定阈值
        threshold = threshold_value
    
    mask = unlabel_mask & (conf >= threshold)
    return mask, threshold