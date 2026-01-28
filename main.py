import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from models.base_gcn import GCN
from models.iceberg import IceBergModel
from utils.data_utils import make_imbalanced

import sys
import io

# 强制标准输出和标准错误流使用 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# --- 参数配置 ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--mode', type=str, 
                    choices=['standard', 'imbalanced', 'weighted', 'iceberg_original', 'iceberg_plus'], 
                    required=True)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--mu', type=float, default=1.0, help='DB调整强度')
parser.add_argument('--lam', type=float, default=0.1, help='无监督损失权重')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 数据加载与预处理 ---
dataset = Planetoid(root='./data', name=args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)

if args.mode != 'standard':
    # 构造不平衡情况，IR=10
    data = make_imbalanced(data, minority_class=0, keep_ratio=0.1)

# --- 模型与优化器初始化 ---
if args.mode == 'iceberg_plus':
    model = IceBergModel(dataset.num_features, 16, dataset.num_classes).to(device)
else:
    model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# --- 训练逻辑 ---
def train(epoch):
    model.train()
    optimizer.zero_grad()
    
    # --- IceBerg 逻辑 (Original vs Plus) ---
    if args.mode in ['iceberg_original', 'iceberg_plus']:
        with torch.no_grad():
            model.eval()
            logits_all = model(data.x, data.edge_index)
            conf, pred_pseudo = F.softmax(logits_all, dim=-1).max(dim=-1)
            unlabel_mask = ~data.train_mask
            pseudo_mask = unlabel_mask & (conf >= conf[unlabel_mask].mean())
            
            # 统计分布 pi_c
            raw_counts = torch.bincount(pred_pseudo[pseudo_mask], minlength=dataset.num_classes).float()
            
            if args.mode == 'iceberg_plus':
                # [增强] 拉普拉斯平滑：+1 确保 pi_c 永不为 0
                pi_c = (raw_counts + 1.0) / (raw_counts.sum() + dataset.num_classes)
            else:
                # [原始] 仅使用基础 epsilon 防止除零
                pi_c = raw_counts / (raw_counts.sum() + 1e-6)

        model.train()
        out = model(data.x, data.edge_index)
        loss_sup = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        
        # 判定是否触发无监督平衡
        # original 模式 warmup_limit 为 0；plus 模式为总 epoch 的 25%
        warmup_limit = (args.epochs // 4) if args.mode == 'iceberg_plus' else 0
        
        if epoch > warmup_limit and pseudo_mask.sum() > 0:
            # 使用 pi_c 进行 Logit 调整
            logits_pseudo_adj = out[pseudo_mask] + args.mu * torch.log(pi_c + 1e-6)
            loss_unsup = F.cross_entropy(logits_pseudo_adj, pred_pseudo[pseudo_mask])
            loss = loss_sup + args.lam * loss_unsup
        else:
            loss = loss_sup
    else:
        # 传统训练逻辑
        weights = None
        if args.mode == 'weighted':
            counts = torch.bincount(data.y[data.train_mask], minlength=dataset.num_classes).float()
            weights = (1.0 / (counts + 1e-6))
            weights /= weights.mean()
            
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=weights)

    loss.backward()
    optimizer.step()
    return loss.item()

# --- 主循环与评价 ---
best_val_acc = 0
final_results = {}

for epoch in range(1, args.epochs + 1):
    loss_val = train(epoch)
    
    # 每轮进行验证
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=-1)
        val_acc = int((pred[data.val_mask] == data.y[data.val_mask]).sum()) / int(data.val_mask.sum())
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 记录测试集表现
            test_mask = data.test_mask
            y_true = data.y[test_mask].cpu().numpy()
            y_pred = pred[test_mask].cpu().numpy()
            cm = confusion_matrix(y_true, y_pred, labels=range(dataset.num_classes))
            final_results = {
                'acc': int((pred[test_mask] == data.y[test_mask]).sum()) / int(test_mask.sum()),
                'f1': f1_score(y_true, y_pred, average='macro'),
                'min_recall': cm[0,0] / (cm[0,:].sum() + 1e-6)
            }

print(f"\nMode: {args.mode} Results:")
print(f"Accuracy: {final_results['acc']:.4f}, Macro-F1: {final_results['f1']:.4f}, Minority Recall: {final_results['min_recall']:.4f}")