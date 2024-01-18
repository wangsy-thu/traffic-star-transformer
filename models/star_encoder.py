from .model_utils.attention import *
from .model_utils.conv import *


class StarEncoderLayer(nn.Module):
    """
    Star Encoder Layer
    Star-Transformer 的编码器
    """

    def __init__(self, K: int, hidden_channels: int, edge_index: torch.LongTensor,
                 vertices_num: int, time_step_num: int, conv_method: str = 'GIN'):
        """
        构造函数
        :param K: GIN网络层数，消息聚集邻居的跳数，默认为 2
        :param hidden_channels: 中间传递的维度
        :param edge_index: 边列表
        :param vertices_num: 节点数量
        :param time_step_num: 时间步数量
        :param conv_method: 卷积策略，默认为 'GIN'
        """
        super(StarEncoderLayer, self).__init__()
        # 时间注意力层
        self.temporal_attention = TemporalMultiHeadAttentionLayer(
            in_channels=hidden_channels,
            vertices_num=vertices_num
        )
        # 空间注意力层
        self.spatial_attention = SpatialMultiHeadAttentionLayer(
            in_channels=hidden_channels,
            time_steps_num=time_step_num
        )
        # 空间卷积层
        self.spatial_conv = SpatialConvLayer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_index=edge_index,
            neighbors_count=K,
            conv_method=conv_method
        )
        # FFN 层，存储元素内部的信息
        self.st_ffn = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=(1, 1),
            stride=(1, 1)
        )
        # 残差卷积层，用于对齐
        self.residual_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=(1, 1),
            stride=(1, 1)
        )
        self.layer_norm = nn.LayerNorm(hidden_channels)

    def forward(self, x_matrix: torch.Tensor, mask=None):
        """
        前向传播函数
        :param mask: 掩码信息
        :param x_matrix: 输入矩阵块 (B, N, C, T)
        :return: 输出矩阵块 (B, N, C, T)
        """
        # 时间注意力
        x_temporal_attn, t_attn = self.temporal_attention(x_matrix, mask)  # (B, N, C, T)

        # 空间注意力
        x_spatial_attn, s_attn = self.spatial_attention(x_temporal_attn)  # (B, N, C, T)

        # 空间卷积
        x_spatial_conv = self.spatial_conv(x_spatial_attn)  # (B, N, C, T)

        # 残差分支
        x_residual = self.residual_conv(x_matrix.permute(0, 2, 1, 3))  # (B, C, N, T)

        # FFN层 from (B, N, C, T) -> (B, C, N, T) -FFN-> (B, C, N, T)
        x_ffn = self.st_ffn(x_spatial_conv.permute(0, 2, 1, 3))

        # Layer Norm from (B, C, N, T) -> (B, N, T, C) -> (B, N, C, T)
        x_out = self.layer_norm(
            F.relu(x_ffn + x_residual).permute(0, 2, 3, 1)
        ).permute(0, 1, 3, 2)

        # 输出与输入统一，便于搭层
        return x_out


class StarEncoder(nn.Module):
    """
    Star Encoder
    Star Transformer 的 Encoder 部分
    """

    def __init__(self, K: int, hidden_channels: int, edge_index: torch.LongTensor,
                 vertices_num: int, time_step_num: int, in_features: int,
                 layer_count: int, conv_method: str = 'GIN'):
        """
        构造函数
        :param K: GIN网络层数，消息聚集邻居的跳数，默认为 2
        :param hidden_channels: 中间传递的维度
        :param edge_index: 边列表
        :param vertices_num: 节点数量
        :param time_step_num: 时间步数量
        :param conv_method: 卷积策略，默认为 'GIN'
        :param in_features: 输入特征数
        :param layer_count: encoder 层数
        """
        super(StarEncoder, self).__init__()
        self.input_conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=hidden_channels,
            kernel_size=(1, 1),
            stride=(1, 1)
        )

        self.encoder_layer_list = nn.ModuleList([
            StarEncoderLayer(
                K=K,
                hidden_channels=hidden_channels,
                edge_index=edge_index,
                vertices_num=vertices_num,
                time_step_num=time_step_num,
                conv_method=conv_method
            ) for _ in range(layer_count)
        ])

    def forward(self, x_matrix: torch.Tensor):
        """
        前向传播函数
        :param x_matrix: _matrix: 输入矩阵块 (B, N, F_in, T)
        :return: 输出矩阵块 (B, N, C, T)
        """

        # 计算 Input Conv 块
        # 动态变化情况 (B, N, F_in, T) -permute-> (B, F_in, N, T)
        # (B, F_in, N, T) -conv-> (B, c_in, N, T) -permute-> (B, N ,c_in, T)
        x_matrix = self.input_conv(
            x_matrix.permute((0, 2, 1, 3))
        ).permute((0, 2, 1, 3))

        # 计算通过 Encoder Layer 块
        for block in self.encoder_layer_list:
            x_matrix = block(x_matrix)  # (B, N, C, T)

        return x_matrix
