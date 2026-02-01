"""
论文 IceBerg §3.2 Propagation then Transformation (PPTT)。
Eq (10) X^(t+1) = (1-α)ÃX^(t) + αX，Eq (11) Ŷ = Softmax(MLP(X^(T)))。
图扩散 + MLP，与 APPNP/D2PT 一致，解耦 P 与 T 以增强长程传播、缓解 few-shot。
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP


class IceBergModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, alpha=0.1):
        super(IceBergModel, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, out_channels)
        )
        self.prop = APPNP(K=K, alpha=alpha)

    def forward(self, x, edge_index):
        x = self.prop(x, edge_index)
        return self.mlp(x)