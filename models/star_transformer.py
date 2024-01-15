import torch
from torch import nn

from models.star_decoder import StarDecoder
from models.star_encoder import StarEncoder
from utils.adj import get_edge_index


class StarTransformer(nn.Module):
    """
    Star Transformer 模型
    """

    def __init__(self, K: int, hidden_channels: int, edge_index: torch.LongTensor,
                 vertices_num: int, time_step_num: int, in_features: int, out_features: int,
                 layer_count: int, predict_step_num: int, conv_method: str = 'GIN'):
        """
        构造函数
        :param K: GIN网络层数，消息聚集邻居的跳数，默认为 2
        :param hidden_channels: 中间传递的维度
        :param edge_index: 边列表
        :param vertices_num: 节点数量
        :param time_step_num: 时间步数量
        :param in_features: 输入特征数
        :param out_features: 输出特征数
        :param layer_count: 层数
        :param predict_step_num: 预测时间步数量
        :param conv_method: 卷积方案，默认为 GIN
        """

        super(StarTransformer, self).__init__()
        self.predict_step_num = predict_step_num
        self.out_features = out_features
        self.encoder = StarEncoder(
            K=K,
            hidden_channels=hidden_channels,
            edge_index=edge_index,
            vertices_num=vertices_num,
            time_step_num=time_step_num,
            in_features=in_features,
            layer_count=layer_count,
            conv_method=conv_method
        )
        self.decoder = StarDecoder(
            K=2,
            hidden_channels=hidden_channels,
            edge_index=edge_index,
            vertices_num=vertices_num,
            time_step_num=predict_step_num,
            out_features=out_features,
            layer_count=layer_count,
            conv_method=conv_method
        )
        self.final_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1)
        )

    def forward(self, x_matrix_in: torch.Tensor, x_matrix_out: torch.Tensor):
        """
        前向传播函数
        :param x_matrix_out: 输出矩阵块 (B, N, F_out, T)
        :param x_matrix_in: 输入矩阵块 (B, N, F_in, T)
        :return: 输出矩阵块 (B, N, F_out, T)
        """
        x_matrix_enc = self.encoder(x_matrix_in)
        x_matrix_dec = self.decoder(x_matrix_out, x_matrix_enc)
        x_out = self.final_conv(x_matrix_dec.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        return x_out

    def inference(self, x_matrix_in: torch):
        """
        自回归预测
        :param x_matrix_in: 输入矩阵块 (B, N, F_in, T)
        :return: 输出矩阵块 (B, N, F_out, T)
        """
        batch_size, vertices_num, in_features, time_steps = x_matrix_in.size()
        model.eval()
        x_matrix_trg = (torch.zeros((batch_size, vertices_num, self.out_features, self.predict_step_num)).
                        to(x_matrix_in.device))
        with torch.no_grad():
            for i in range(self.predict_step_num):
                x_out = model(x_matrix_in, x_matrix_trg)
                x_matrix_trg[:, :, :, i] = x_out[:, :, :, i]
                x_matrix_trg.contiguous()
        return x_matrix_trg


def make_model(neighbor_count: int, hidden_channels: int, edge_index: torch.LongTensor,
               vertices_num: int, time_step_num: int, in_features: int, out_features: int,
               device: str, layer_count: int, predict_step_num: int, conv_method: str = 'GIN'):
    model = StarTransformer(
        K=neighbor_count,
        hidden_channels=hidden_channels,
        edge_index=torch.from_numpy(edge_index).type(torch.long).to('cuda'),
        vertices_num=vertices_num,
        time_step_num=time_step_num,
        in_features=in_features,
        out_features=out_features,
        predict_step_num=predict_step_num,
        layer_count=layer_count,
        conv_method=conv_method
    )
    model.to(device)
    return model


if __name__ == '__main__':
    data_dir = '../data/PEMS04/PEMS04.csv'
    edge_index = get_edge_index(data_dir)
    in_features = 3
    out_features = 1
    hidden_channels = 64
    batch_size = 32
    vertices_num = 374
    time_step_num = 12
    predict_step_num = 24

    # model = StarEncoderLayer(
    #     K=2,
    #     hidden_channels=out_channels,
    #     edge_index=torch.from_numpy(edge_index).type(torch.long).to('cuda'),
    #     vertices_num=vertices_num,
    #     time_step_num=time_step_num, conv_method='GIN'
    # )

    model = make_model(
        neighbor_count=2,
        hidden_channels=hidden_channels,
        edge_index=edge_index,
        vertices_num=vertices_num,
        time_step_num=time_step_num,
        in_features=in_features,
        out_features=out_features,
        predict_step_num=predict_step_num,
        layer_count=2,
        conv_method='GIN',
        device='cuda'
    )

    X = torch.rand((batch_size, vertices_num, in_features, time_step_num)).to('cuda')
    Y = torch.rand((batch_size, vertices_num, out_features, predict_step_num)).to('cuda')

    print('input X shape: {}'.format(X.size()))
    print('input Y shape: {}'.format(Y.size()))
    out = model(X, Y)
    out_inf = model.inference(X)
    print('output O shape: {}'.format(out.size()))
    print('output I shape: {}'.format(out_inf.size()))
