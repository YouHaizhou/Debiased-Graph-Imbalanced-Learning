import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP

class IceBergModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, alpha=0.1):
        super(IceBergModel, self).__init__()
        # Transformation (T)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        # Propagation (P) - PPTT架构核心
        self.prop = APPNP(K=K, alpha=alpha)

    def forward(self, x, edge_index):
        # 先传播捕获结构信号，再进行特征变换
        x = self.prop(x, edge_index)
        return self.mlp(x)