import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as gnn

from utils.adj import get_edge_index


def build_gnn_layer(conv_method: str, in_channels: int, out_channels: int):
    if conv_method == 'GIN':
        return gnn.GINConv(
            nn=nn.Sequential(
                nn.Linear(in_channels, 4 * in_channels),
                nn.ReLU(),
                nn.Linear(4 * in_channels, out_channels)
            )
        )
    elif conv_method == 'GCN':
        return gnn.GCNConv(
            in_channels=in_channels,
            out_channels=out_channels
        )
    elif conv_method == 'SAGE':
        return gnn.SAGEConv(
            in_channels=in_channels,
            out_channels=out_channels
        )
    elif conv_method == 'GAT':
        return gnn.GATConv(
            in_channels=in_channels,
            out_channels=out_channels
        )
    else:
        return gnn.GINConv(
            nn=nn.Sequential(
                nn.Linear(in_channels, 4 * in_channels),
                nn.ReLU(),
                nn.Linear(4 * in_channels, out_channels)
            )
        )


class GNNConvLayer(nn.Module):
    """
    基于 GNN 的空间卷积层
    """

    def __init__(self, in_channels: int, out_channels: int, edge_index: torch.Tensor,
                 neighbors_count=2, conv_method='GIN'):
        """
        构造函数
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param edge_index: 边列表
        :param neighbors_count: 聚集邻居数量
        """
        super(GNNConvLayer, self).__init__()
        assert conv_method in ['GIN', 'GCN', 'SAGE', 'TCN', 'GAT']
        self.K = neighbors_count
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gnn_layers = nn.ModuleList([
            build_gnn_layer(conv_method, in_channels, out_channels)
        ])
        self.gnn_layers.extend([
            build_gnn_layer(
                conv_method, out_channels, out_channels
            ) for _ in range(neighbors_count - 1)
        ])
        self.edge_index = edge_index

    def forward(self, graph_sig: torch.Tensor):
        """
        正向传播函数
        :param graph_sig: 图信号 (B, N, C_in)
        :return: 图卷积后的数据 (B, N, C_out)
        """
        for gnn_layer in self.gnn_layers:
            graph_sig = gnn_layer(graph_sig, self.edge_index)
        return graph_sig


class SpatialConvLayer(nn.Module):
    """
    空间卷积层 Spatial Convolution Layer
    """
    def __init__(self, in_channels: int, out_channels: int, edge_index: torch.LongTensor,
                 neighbors_count: int = 2, conv_method: str = 'GIN'):
        super(SpatialConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_index = edge_index
        self.K = neighbors_count
        self.gnn_conv = GNNConvLayer(
            in_channels, out_channels, edge_index, neighbors_count, conv_method
        )

    def forward(self, x_matrix: torch.Tensor):
        """
        前向传播函数
        :param x_matrix: 输入矩阵数据 (B, N, C_in, T)
        :return: 输出矩阵数据 (B, N, C_out, T)
        """
        batch_size, vertices_num, in_channel, time_step_num = x_matrix.size()
        outputs = []

        for time_step in range(time_step_num):
            # 提取该时间步下的图信号
            graph_sig = x_matrix[:, :, :, time_step]
            # 自定义的中间变量，不参与模型的梯度更新，
            output = self.gnn_conv(graph_sig)
            # 堆叠输出
            outputs.append(output.unsqueeze(-1))  # (B, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))


if __name__ == '__main__':
    data_dir = '../../data/PEMS04/PEMS04.csv'
    edge_index = get_edge_index(data_dir)
    in_channels = 3
    out_channels = 512
    batch_size = 64
    vertices_num = 374
    time_step_num = 12

    # model = GNNConvLayer(
    #     in_channels=in_channels,
    #     out_channels=out_channels,
    #     edge_index=torch.from_numpy(edge_index).type(torch.long).to('cpu')
    # )
    #
    # X = torch.rand((batch_size, vertices_num, in_channels))
    # print('input X shape: {}'.format(X.size()))
    # Y = model(X)
    # print('output X shape: {}'.format(Y.size()))

    model = SpatialConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        edge_index=torch.from_numpy(edge_index).type(torch.long).to('cpu'),
        neighbors_count=2,
        conv_method='GIN'
    )

    X = torch.rand((batch_size, vertices_num, in_channels, time_step_num))
    print('input X shape: {}'.format(X.size()))
    Y = model(X)
    print('output X shape: {}'.format(Y.size()))
