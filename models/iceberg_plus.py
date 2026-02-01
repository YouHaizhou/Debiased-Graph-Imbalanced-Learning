from .iceberg import IceBergModel

class IceBergPlusModel(IceBergModel):
    """
    IceBerg+ 基线模型。
    继承自 IceBergModel 的 PPTT (Propagation then Transformation) 架构。
    """
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, alpha=0.1):
        super(IceBergPlusModel, self).__init__(in_channels, hidden_channels, out_channels, K, alpha)