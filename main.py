import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from models.gcn import GCN
from models.iceberg import IceBergModel
from models.iceberg_plus import IceBergPlusModel
from utils.threshold_utils import get_threshold_mask
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
#置信度策略
parser.add_argument('--threshold_strategy', type=str, default='dynamic_mean',
                    choices=['dynamic_mean', 'global_mean', 'fixed'])
parser.add_argument('--fixed_threshold', type=float, default=0.7)
parser.add_argument('--diagnose', action='store_true', help='打印伪标签分布、pi_c、少数类logit等诊断信息')
# Noise-Tolerant Double Balancing (论文 Eq 9)，仅 iceberg_original 时生效
parser.add_argument('--beta', type=float, default=0.0, help='NT-DB 对称项权重，>0 时 loss_unsup = (1+beta)*CE')

args = parser.parse_args()

# 诊断用：少数类类别（与 make_imbalanced 一致）
MINORITY_CLASS = 0
# 诊断采样 epoch（首轮、warmup 后、中期、末轮）
_warmup_limit = (args.epochs // 4) if args.mode == 'iceberg_plus' else 0
DIAG_EPOCHS = sorted(set([1, max(1, _warmup_limit + 1), args.epochs // 2, args.epochs]))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 数据加载与预处理 ---
dataset = Planetoid(root='./data', name=args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)

if args.mode != 'standard':
    # 构造不平衡情况，IR=10
    data = make_imbalanced(data, minority_class=0, keep_ratio=0.1)

# --- 模型与优化器初始化 ---
if args.mode == 'iceberg_plus':
    model = IceBergPlusModel(dataset.num_features, 16, dataset.num_classes).to(device)
elif args.mode == 'iceberg_original':
    model = IceBergModel(dataset.num_features, 16, dataset.num_classes).to(device)
else:
    model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# --- 训练逻辑 ---
def train(epoch):
    model.train()
    optimizer.zero_grad()
    
    if args.mode in ['iceberg_original', 'iceberg_plus']:
        with torch.no_grad():
            model.eval()
            logits_all = model(data.x, data.edge_index)
            probs = torch.softmax(logits_all, dim=-1)
            conf, pred_pseudo = probs.max(dim=-1)
            # 论文 Algorithm 1: get_confidence(logits) -> confidence = max(softmax), pred = argmax
            # 论文 Eq (5): 动态阈值 τ' = (1/|V_U|) Σ_j max(f_θ(v_j))
            strategy = 'dynamic_mean' if args.mode == 'iceberg_original' else args.threshold_strategy
            pseudo_mask, _ = get_threshold_mask(
                conf, ~data.train_mask,
                strategy=strategy,
                threshold_value=args.fixed_threshold
            )
            # 论文 Eq (6) + Algorithm 1: 伪标签类别计数，再归一化为分布 π_c（用于 Eq 8 balanced softmax）
            raw_counts = torch.bincount(pred_pseudo[pseudo_mask], minlength=dataset.num_classes).float()
            if args.mode == 'iceberg_plus':
                pi_c = (raw_counts + 1.0) / (raw_counts.sum() + dataset.num_classes)
            else:
                # 论文：π_c 为伪标签类比例；零计数时 log(π_c+ε) 保持数值稳定
                pi_c = raw_counts / (raw_counts.sum() + 1e-6)

            # 诊断：伪标签分布、pi_c、少数类 logit（在指定 epoch 打印）
            if getattr(args, 'diagnose', False) and epoch in DIAG_EPOCHS:
                n_pseudo = int(pseudo_mask.sum().item())
                pseudo_dist = (raw_counts / (raw_counts.sum() + 1e-6)).cpu().tolist()
                pi_c_list = pi_c.cpu().tolist()
                test_minority_mask = data.test_mask & (data.y == MINORITY_CLASS)
                if test_minority_mask.sum() > 0:
                    min_logits = logits_all[test_minority_mask, MINORITY_CLASS].cpu()
                    other_max = logits_all[test_minority_mask].cpu().clone()
                    other_max[:, MINORITY_CLASS] = -1e9
                    other_max = other_max.max(dim=1).values
                    print(f"[Diagnose Epoch {epoch}] Pseudo#={n_pseudo} | "
                          f"Pseudo_dist={[round(x, 4) for x in pseudo_dist]} | "
                          f"pi_c={[round(x, 4) for x in pi_c_list]}")
                    print(f"  Test minority nodes (n={int(test_minority_mask.sum())}): "
                          f"minority_logit mean={min_logits.mean():.4f} min={min_logits.min():.4f} max={min_logits.max():.4f} | "
                          f"other_max_logit mean={other_max.mean():.4f} (minority<other -> 预测多数类)")

        model.train()
        out = model(data.x, data.edge_index)
        loss_sup = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        
        # 判定是否触发无监督平衡
        # original 模式 warmup_limit 为 0；plus 模式为总 epoch 的 25%
        warmup_limit = (args.epochs // 4) if args.mode == 'iceberg_plus' else 0
        
        if epoch > warmup_limit and pseudo_mask.sum() > 0:
            # 论文 Eq (8): q_j[c] = f_θ(v_j)[c] + μ·log π_c；Eq (7): L_unsup = -λ·E[ I(≥τ') ℓ(q_j, ŷ_j) ]
            logits_pseudo_adj = out[pseudo_mask] + args.mu * torch.log(pi_c + 1e-6)
            loss_unsup = F.cross_entropy(logits_pseudo_adj, pred_pseudo[pseudo_mask])
            # 论文 Eq (9) Noise-Tolerant Double Balancing: 对称项 β·ℓ(ŷ_j, max(q_j))，此处实现为 (1+β)*CE
            if args.mode == 'iceberg_original' and getattr(args, 'beta', 0.0) > 0:
                loss_unsup = (1.0 + args.beta) * loss_unsup
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