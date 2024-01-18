import torch

from models.star_decoder import StarDecoder
from models.star_encoder import StarEncoder
from models.star_transformer import make_model
from utils.adj import get_edge_index


def test_encoder():
    data_dir = '../data/PEMS04/PEMS04.csv'
    edge_index = get_edge_index(data_dir)
    in_channels = 3
    out_channels = 64
    batch_size = 32
    vertices_num = 374
    time_step_num = 12

    # model = StarEncoderLayer(
    #     K=2,
    #     hidden_channels=out_channels,
    #     edge_index=torch.from_numpy(edge_index).type(torch.long).to('cuda'),
    #     vertices_num=vertices_num,
    #     time_step_num=time_step_num, conv_method='GIN'
    # )

    model = StarEncoder(
        K=2,
        hidden_channels=out_channels,
        edge_index=torch.from_numpy(edge_index).type(torch.long).to('cuda'),
        vertices_num=vertices_num,
        time_step_num=time_step_num,
        in_features=in_channels,
        layer_count=2,
        conv_method='GCN'
    )

    X = torch.rand((batch_size, vertices_num, in_channels, time_step_num)).to('cuda')
    model.to('cuda')
    print('input X shape: {}'.format(X.size()))
    Y = model(X)
    print('output X shape: {}'.format(Y.size()))


def test_decoder():
    hidden_channels = 64
    out_channels = 1
    in_step_num = 12
    batch_size = 32
    vertices_num = 374
    out_step_num = 24

    # model = StarEncoderLayer(
    #     K=2,
    #     hidden_channels=out_channels,
    #     edge_index=torch.from_numpy(edge_index).type(torch.long).to('cuda'),
    #     vertices_num=vertices_num,
    #     time_step_num=time_step_num, conv_method='GIN'
    # )

    model = StarDecoder(
        hidden_channels=hidden_channels,
        vertices_num=vertices_num,
        out_features=out_channels,
        layer_count=2,
    )

    X1 = torch.rand((batch_size, vertices_num, out_channels, 1)).to('cuda')
    X0 = torch.rand((batch_size, vertices_num, out_channels, out_step_num)).to('cuda')
    X2 = torch.rand((batch_size, vertices_num, hidden_channels, in_step_num)).to('cuda')
    model.to('cuda')
    print('input X1 shape: {}'.format(X1.size()))
    print('input X2 shape: {}'.format(X2.size()))
    Y1 = model(X1, X2)
    Y2 = model(X0, X2)
    print('output Y1 shape: {}'.format(Y1.size()))
    print('output Y2 shape: {}'.format(Y2.size()))


def test_transformer():
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
