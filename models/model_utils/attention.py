import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.nn import functional as F


class SpatialAttentionLayer(nn.Module):
    """
    Spatial Attention Layer
    空间注意力层，计算 N 个节点的注意力矩阵
    """

    def __init__(self, in_channels: int, vertices_num: int, time_steps_num: int):
        """
        构造函数
        :param in_channels: 输入通道数，特征数， Integer 类型
        :param vertices_num: 图节点数量， Integer 类型
        :param time_steps_num: 时间步数量，指的是输入的时间段的个数， Integer 类型
        """
        super(SpatialAttentionLayer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(time_steps_num))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, time_steps_num))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.bs = nn.Parameter(torch.FloatTensor(1, vertices_num, vertices_num))
        self.Vs = nn.Parameter(torch.FloatTensor(vertices_num, vertices_num))

    def forward(self, x_matrix):
        """
        前向传播函数
        :param x_matrix: 输入数据块 (B, N, F_in, T)
        :return: 输出为空间注意力矩阵 (B, N, N)
        """
        # 消除特征维度 F
        mat_n_l = torch.matmul(torch.matmul(x_matrix, self.W1), self.W2)  # (B, N, T)
        # 另外一半消除特征维度 F
        mat_n_r = torch.matmul(self.W3, x_matrix).transpose(-1, -2)  # (B, T, N)
        dot_product = torch.matmul(mat_n_l, mat_n_r)  # (B, N, N)
        # 对 dot_product Attention 做线性重组

        S_att = torch.matmul(self.Vs, torch.sigmoid(dot_product + self.bs))
        # 做 Softmax 归一化，使得注意力输出每行和为 1
        S_norm = F.softmax(S_att, dim=1)
        return S_norm


class TemporalAttentionLayer(nn.Module):
    """
    时间注意力机层，计算所有时间步的注意力矩阵
    """

    def __init__(self, in_channels: int, vertices_num: int, time_steps_num: int):
        """
        构造函数
        :param in_channels: 输入通道数，特征数， Integer 类型
        :param vertices_num: 图节点数量， Integer 类型
        :param time_steps_num: 时间步数量，指的是输入的时间段的个数， Integer 类型
        """
        super(TemporalAttentionLayer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(vertices_num))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, vertices_num))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.be = nn.Parameter(torch.FloatTensor(1, time_steps_num, time_steps_num))
        self.Ve = nn.Parameter(torch.FloatTensor(time_steps_num, time_steps_num))

    def forward(self, x_matrix, mask=None):
        """
        前向传播函数
        :param x_matrix: 输入数据块 (B, N, C, T)
        :param mask: 掩码矩阵 (B, T, T)
        :return: 输出为时间注意力矩阵 (B, T, T)
        """
        # 消除特征维度 F (B, T, N)
        mat_t_l = torch.matmul(torch.matmul(x_matrix.permute(0, 3, 2, 1), self.U1), self.U2)
        # 右侧消除特征维度 F (B, N, T)
        mat_t_r = torch.matmul(self.U3, x_matrix)
        dot_product = torch.matmul(mat_t_l, mat_t_r)  # (B, T, T)
        # 线性重分布
        E_att = torch.matmul(self.Ve, torch.sigmoid(dot_product + self.be))  # (B, T, T)
        if mask is not None:
            E_att = E_att.masked_fill(mask, -1e6)
        E_norm = F.softmax(E_att, dim=1)
        return E_norm


class SpatialMultiHeadAttentionLayer(nn.Module):
    """
    Spatial Multi-Head Attention Layer
    空间注意力机制，计算 N 个节点的注意力矩阵以及输出结果
    """

    def __init__(self, in_channels: int, time_steps_num: int, heads_num=1):
        """
        构造函数
        :param in_channels: 输入通道数，特征数， Integer 类型
        :param time_steps_num: 时间步数量，指的是输入的时间段的个数， Integer 类型
        :param heads_num: 注意力头数量
        """
        assert time_steps_num % heads_num == 0
        super(SpatialMultiHeadAttentionLayer, self).__init__()
        # 空间维度的自注意力机制
        self.self_attention = MultiheadAttention(
            embed_dim=time_steps_num,
            num_heads=heads_num,
            batch_first=True
        )
        # 消除特征维度
        self.channel_linear = nn.Linear(in_features=in_channels, out_features=1)

    def forward(self, x_matrix: torch.Tensor):
        """
        前向传播函数
        :param x_matrix: 输入数据块 (B, N, C, T)
        :return: 输出经过空间注意力计算的数据块 (B, N, C, T)
        """
        batch_size, vertices_num, channel_num, time_step_num = x_matrix.size()
        x_mat_in = self.channel_linear(x_matrix.permute((0, 1, 3, 2))).squeeze()
        _, spatial_attention = self.self_attention(
            query=x_mat_in,
            key=x_mat_in,
            value=x_mat_in
        )
        x_mat_out = torch.matmul(
            x_matrix.permute((0, 2, 3, 1)).reshape(batch_size, -1, vertices_num),
            spatial_attention.transpose(-1, -2)
        )
        x_mat_out = (x_mat_out
                     .reshape(batch_size, time_step_num, channel_num, vertices_num)
                     .permute((0, 3, 2, 1)))
        return x_mat_out, spatial_attention


class TemporalMultiHeadAttentionLayer(nn.Module):
    """
    Temporal Multi-Head Attention Layer
    时间注意力机制，计算 N 个节点的注意力矩阵以及输出结果
    """

    def __init__(self, in_channels: int, vertices_num: int, heads_num=1):
        """
        构造函数
        :param in_channels: 输入通道数，特征数， Integer 类型
        :param vertices_num: 图节点数量， Integer 类型
        :param heads_num: 注意力头数量
        """
        assert vertices_num % heads_num == 0
        super(TemporalMultiHeadAttentionLayer, self).__init__()
        # 空间维度的自注意力机制
        self.self_attention = MultiheadAttention(
            embed_dim=vertices_num,
            num_heads=heads_num,
            batch_first=True
        )
        # 消除特征维度
        self.channel_linear = nn.Linear(in_features=in_channels, out_features=1)

    def forward(self, x_matrix: torch.Tensor, mask=None):
        """
        前向传播函数
        :param x_matrix: 输入数据块 (B, N, C, T)
        :param mask: 掩码矩阵
        :return: 输出经过时间注意力计算的数据块 (B, N, C, T)
        """
        x_mat_in = self.channel_linear(x_matrix.permute((0, 3, 1, 2)))
        x_mat_in = torch.squeeze(x_mat_in, dim=3)
        _, temporal_attention = self.self_attention(
            query=x_mat_in,
            key=x_mat_in,
            value=x_mat_in,
            attn_mask=mask
        )
        batch_size, vertices_num, channel_num, time_step_num = x_matrix.size()
        x_mat_out = torch.matmul(x_matrix.reshape(batch_size, -1, time_step_num),
                                 temporal_attention.transpose(-1, -2))
        x_mat_out = x_mat_out.reshape(batch_size, vertices_num, channel_num, time_step_num)
        return x_mat_out, temporal_attention


class TemporalCrossAttentionLayer(nn.Module):
    """
    Temporal Cross Attention Layer
    时间互注意力机制，计算 N 个节点的注意力矩阵以及输出结果
    """
    def __init__(self, in_channels: int, vertices_num: int, heads_num=1):
        """
        构造函数
        :param in_channels: 输入通道数，特征数， Integer 类型
        :param vertices_num: 图节点数量， Integer 类型
        :param heads_num: 注意力头数量
        """
        assert vertices_num % heads_num == 0
        super(TemporalCrossAttentionLayer, self).__init__()
        # 空间维度的互注意力机制
        self.cross_attention = MultiheadAttention(
            embed_dim=vertices_num,
            num_heads=heads_num,
            batch_first=True
        )
        # 消除特征维度
        self.channel_input_linear = nn.Linear(in_features=in_channels, out_features=1)
        self.channel_output_linear = nn.Linear(in_features=in_channels, out_features=1)

    def forward(self, x_matrix_in: torch.Tensor, x_matrix_out: torch.Tensor, mask=None):
        """
        前向传播函数
        :param x_matrix_out: 输入数据块 (B, N, C, T)
        :param x_matrix_in: 输入数据块 (B, N, C, T)
        :param mask: 掩码矩阵
        :return: 输出经过时间注意力计算的数据块 (B, N, C, T)
        """
        x_mat_in = self.channel_input_linear(x_matrix_in.permute((0, 3, 1, 2)))
        x_mat_in = torch.squeeze(x_mat_in, dim=3)
        x_mat_out = self.channel_output_linear(x_matrix_out.permute((0, 3, 1, 2)))
        x_mat_out = torch.squeeze(x_mat_out, dim=3)
        _, temporal_attention = self.cross_attention(
            query=x_mat_in,
            key=x_mat_out,
            value=x_mat_out,
            attn_mask=mask
        )
        batch_size, vertices_num, channel_num, time_step_num = x_matrix_out.size()
        time_step_out = x_matrix_in.size(-1)
        x_out = torch.matmul(x_matrix_out.reshape(batch_size, -1, time_step_num),
                             temporal_attention.transpose(-1, -2))
        x_out = x_out.reshape(batch_size, vertices_num, channel_num, time_step_out)
        return x_out, temporal_attention


if __name__ == '__main__':
    vertices_num = 307
    features_num = 10
    in_channels = 3
    time_step_num = 12
    time_step_out = 24
    batch_size = 64

    X1 = torch.rand((batch_size, vertices_num, features_num, time_step_num))
    X2 = torch.rand((batch_size, vertices_num, features_num, time_step_out))
    print('input X1 shape: {}'.format(X1.size()))
    print('input X2 shape: {}'.format(X2.size()))

    # model = SpatialMultiHeadAttentionLayer(
    #     in_channels=features_num,
    #     time_steps_num=time_step_num,
    #     heads_num=1
    # )
    # model = TemporalMultiHeadAttentionLayer(
    #     in_channels=features_num,
    #     vertices_num=vertices_num,
    #     heads_num=1
    # )
    model = TemporalCrossAttentionLayer(
        in_channels=features_num,
        vertices_num=vertices_num,
        heads_num=1
    )
    output_x, attn = model(X1, X2)

    print('output X shape: {}'.format(output_x.size()))
    print('output Attention shape: {}'.format(attn.size()))
