from .model_utils.attention import *
from .model_utils.conv import *


class StarDecoderLayer(nn.Module):
    """
    Star Decoder Layer
    Star-Transformer 的解码器
    """

    def __init__(self, hidden_channels: int, vertices_num: int):
        """
        构造函数
        :param hidden_channels: 中间传递的维度
        :param vertices_num: 节点数量
        """
        super(StarDecoderLayer, self).__init__()
        # 时间注意力层
        self.temporal_attention = TemporalMultiHeadAttentionLayer(
            in_channels=hidden_channels,
            vertices_num=vertices_num
        )
        # 时间互注意力层
        self.temporal_cross_attention = TemporalCrossAttentionLayer(
            in_channels=hidden_channels,
            vertices_num=vertices_num
        )
        # FFN 层，存储元素内部的信息
        self.st_ffn = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=(1, 1),
            stride=(1, 1)
        )
        self.layer_norm = nn.LayerNorm(hidden_channels)

    def forward(self, x_matrix_in: torch.Tensor, x_matrix_out: torch.Tensor, mask=None):
        """
        前向传播函数
        :param x_matrix_in: 输入矩阵块 (B, N, C, T)
        :param x_matrix_out: 输入矩阵块 (B, N, C, T)
        :param mask: 掩码信息
        :return: 输出矩阵块 (B, N, C, T)
        """
        # 时间注意力
        time_step_num = x_matrix_in.size(-1)
        temporal_mask = (~torch.tril(torch.ones(time_step_num, time_step_num), diagonal=0).bool().
                         to(x_matrix_in.device))
        x_temporal_attn, t_attn = self.temporal_attention(x_matrix_in, temporal_mask)  # (B, N, C, T)

        # 层归一化
        x_temporal_attn = self.layer_norm(
            F.relu(x_temporal_attn + x_matrix_in).permute(0, 1, 3, 2)
        ).permute(0, 1, 3, 2)

        x_temporal_cross_attn, cross_attn = self.temporal_cross_attention(
            x_temporal_attn, x_matrix_out, mask
        )

        # 层归一化
        x_temporal_cross_attn = self.layer_norm(
            F.relu(x_temporal_cross_attn + x_temporal_attn).permute(0, 1, 3, 2)
        ).permute(0, 1, 3, 2)

        # FFN层 from (B, N, C, T) -> (B, C, N, T) -FFN-> (B, N, C, T)
        x_ffn = self.st_ffn(x_temporal_cross_attn.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        # Layer Norm from (B, N, C, T) -> (B, N, T, C) -> (B, N, C, T)
        x_out = self.layer_norm(
            F.relu(x_ffn + x_temporal_cross_attn).permute(0, 1, 3, 2)
        ).permute(0, 1, 3, 2)

        # 输出与输入统一，便于搭层
        return x_out


class StarDecoder(nn.Module):
    """
    Star Decoder
    Star Transformer 的 Decoder 部分
    """

    def __init__(self, hidden_channels: int, vertices_num: int,
                 out_features: int, layer_count: int):
        """
        构造函数
        :param hidden_channels: 中间传递的维度
        :param vertices_num: 节点数量
        :param out_features: 输出特征数
        :param layer_count: encoder 层数
        """
        super(StarDecoder, self).__init__()
        self.output_conv = nn.Conv2d(
            in_channels=out_features,
            out_channels=hidden_channels,
            kernel_size=(1, 1),
            stride=(1, 1)
        )

        self.encoder_layer_list = nn.ModuleList([
            StarDecoderLayer(
                hidden_channels=hidden_channels,
                vertices_num=vertices_num,
            ) for _ in range(layer_count)
        ])

        self.final_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1)
        )

    def forward(self, x_matrix_in: torch.Tensor, x_matrix_out: torch.Tensor):
        """
        前向传播函数
        :param x_matrix_out: 输入矩阵块 (B, N, F_in, T)
        :param x_matrix_in: 输入矩阵块 (B, N, F_in, T)
        :return: 输出矩阵块 (B, N, C, T)
        """

        # 计算 Input Conv 块
        # 动态变化情况 (B, N, F_in, T) -permute-> (B, F_in, N, T)
        # (B, F_in, N, T) -conv-> (B, c_in, N, T) -permute-> (B, N ,c_in, T)
        x_matrix = self.output_conv(
            x_matrix_in.permute((0, 2, 1, 3))
        ).permute((0, 2, 1, 3))

        # 计算通过 Encoder Layer 块
        for block in self.encoder_layer_list:
            x_matrix = block(x_matrix, x_matrix_out)  # (B, N, C, T)

        x_matrix = self.final_conv(x_matrix.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        return x_matrix
