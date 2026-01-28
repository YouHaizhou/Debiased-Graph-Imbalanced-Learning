# utils/data_utils.py
import torch

def make_imbalanced(data, minority_class=0, keep_ratio=0.1):
    """
    构造不平衡数据集的工具函数
    """
    new_data = data.clone()
    train_indices = new_data.train_mask.nonzero(as_tuple=False).view(-1)
    labels = new_data.y[train_indices]
    minority_indices = train_indices[labels == minority_class]
    
    num_keep = max(1, int(len(minority_indices) * keep_ratio))
    num_remove = len(minority_indices) - num_keep
    
    perm = torch.randperm(len(minority_indices))
    remove_indices = minority_indices[perm[:num_remove]]
    new_data.train_mask[remove_indices] = False
    
    return new_data