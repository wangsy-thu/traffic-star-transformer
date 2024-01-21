import argparse
import configparser
import os
import shutil

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.make_data_loader import make_flow_data_loader
from models.star_transformer import make_model
from utils.adj import get_edge_index
from utils.inference import predict_and_save_results
from utils.loss import compute_val_loss
from utils.metrics import masked_mae, masked_mse

# 1,解析参数与配置文件
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='config/PEMS04_star.conf', type=str,
                    help="configuration file path")
parser.add_argument("--predict", default=False, type=bool, help="only predict")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % args.config)
config.read(args.config)
only_predict = args.predict
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
time_step_num = int(data_config['time_step_num'])
dataset_name = data_config['dataset_name']

model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print("Cuda Available:{}, use {}!".format(USE_CUDA, DEVICE))

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = num_of_hours
in_features = int(training_config['in_features'])
out_features = int(training_config['out_features'])
hidden_channel = int(training_config['hidden_channel'])
graph_conv_method = training_config['graph_conv_method']
K = int(training_config['K'])
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])
layer_num = int(training_config['layer_num'])
train_device = training_config['train_device']
plot_every = int(training_config['plot_every'])

folder_dir = '%s_%d_channel%d_%e_%s' \
             % (model_name, layer_num, hidden_channel, learning_rate, graph_conv_method)
params_path = os.path.join('./experiments', dataset_name, folder_dir)

edge_index = get_edge_index(adj_filename)
print('batch size: {}'.format(batch_size))
sw = None
if only_predict:
    sw = SummaryWriter(params_path, flush_secs=5)


# 2,初始化模型与数据集
train_loader, val_loader, test_loader, test_target_tensor, _mean, _std = make_flow_data_loader(
    flow_matrix_filename=graph_signal_matrix_filename,
    num_of_weeks=num_of_weeks,
    num_of_days=num_of_days,
    num_of_hours=num_of_hours,
    batch_size=batch_size,
    device=DEVICE,
    shuffle=True
)

model = make_model(
    neighbor_count=K,
    hidden_channels=hidden_channel,
    edge_index=edge_index,
    vertices_num=num_of_vertices,
    time_step_num=time_step_num,
    in_features=in_features,
    out_features=out_features,
    predict_step_num=num_for_predict,
    layer_count=layer_num,
    conv_method=graph_conv_method,
    device=train_device
)


def train_main():
    """
    Train the Model
    """

    # 1, 解析训练参数
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        # 从头训练且不存在参数文件夹 -> 创建参数文件夹
        os.makedirs(params_path)
        print('Create params directory {}'.format(params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        # 从头训练且存在参数文件夹 -> 先删除，后创建
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('Delete Old directory, Create params directory {}'.format(params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        # 断点续训 -> 后续选择
        print('Train from params directory')
    else:
        # 其余任何情况均为非法
        raise SystemExit('Wrong Hyper Params')
    sw = SummaryWriter(params_path, flush_secs=5)

    # 2,定义训练要素: Loss, Optimizer, Model
    criterion = nn.MSELoss().to(DEVICE)  # MSE 损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器
    criterion_masked = masked_mae
    masked_flag = 0
    if loss_function == 'masked_mse':
        criterion_masked = masked_mse  # nn.MSELoss().to(DEVICE)
        masked_flag = 1
    elif loss_function == 'masked_mae':
        criterion_masked = masked_mae
        masked_flag = 1
    elif loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function == 'rmse':
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag = 0

    # 3, 加载断点续训模型
    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    if start_epoch > 0:
        params_file_name = os.path.join(params_path, 'best_model.params')
        model.load_state_dict(torch.load(params_file_name, map_location=DEVICE))
        print('Start Epoch: {}'.format(start_epoch))

    # 4,训练模型
    print('=====Start Training=====')
    for epoch in range(start_epoch, epochs):

        # 1,判断是否保存上次训练后的模型
        params_file_name = os.path.join(params_path, 'best_model.params')
        if masked_flag:
            val_loss = compute_val_loss(
                net=model,
                val_loader=val_loader,
                device=train_device,
                plot_every=plot_every,
                criterion=criterion_masked,
                masked_flag=masked_flag,
                missing_value=missing_value,
                sw=sw,
                epoch=epoch
            )
        else:
            val_loss = compute_val_loss(
                net=model,
                val_loader=val_loader,
                criterion=criterion,
                device=train_device,
                plot_every=plot_every,
                masked_flag=masked_flag,
                missing_value=missing_value,
                sw=sw,
                epoch=epoch
            )

        # 找到最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), params_file_name)
            print('Save Model to: {}'.format(params_path))

        model.train()  # 恢复训练模式，开启梯度更新
        print('=====Epoch [{}]====='.format(epoch + 1))
        for batch_data in tqdm(train_loader, desc='Epoch: {}'.format(epoch + 1)):
            encoder_input, labels = batch_data
            labels = labels.unsqueeze(2)
            current_batch_size = labels.size(0)

            # 区别于 token 预测，这里需要一个启动子
            start_engine = -torch.ones((current_batch_size, num_of_vertices, out_features, 1)).to(train_device)
            labels_input = torch.cat([start_engine, labels], dim=-1)

            # 训练模型
            optimizer.zero_grad()  # 梯度清零
            outputs = model(encoder_input, labels_input)  # 正向传播

            # 计算损失
            if masked_flag:
                loss = criterion_masked(outputs[:, :, :, :-1], labels, missing_value)
            else:
                loss = criterion(outputs[:, :, :, :-1], labels)

            loss.backward()  # 反向求导
            optimizer.step()  # 梯度更新

            train_loss = loss.item()
            global_step += 1
            sw.add_scalar('training_loss', train_loss, global_step)

    print('=====Train Finished=====')
    print('=====Best Epoch: {}====='.format(best_epoch))
    print('=====Predict Start=====')
    predict_main(
        global_step=best_epoch,
        data_loader=test_loader,
        test_target_tensor=test_target_tensor,
        metric_method=metric_method,
        _mean=_mean,
        _std=_std,
        run_type='test',
        sw=sw
    )
    print('=====Predict Success=====')


def predict_main(global_step: int, data_loader: DataLoader,
                 test_target_tensor: torch.Tensor,
                 metric_method: str, _mean, _std, run_type: str,
                 sw):
    """
    Predict 预测模型搭建
    :param sw: 日志工具
    :param global_step: 使用哪步作为参数
    :param data_loader: 数据加载器
    :param test_target_tensor: 标签矩阵
    :param metric_method: 评价标准方法
    :param _mean: 均值
    :param _std: 方差
    :param run_type: 类型
    """

    params_file_name = os.path.join(params_path, 'best_model.params')
    print('Loading Model From Step: {}'.format(global_step))
    model.load_state_dict(torch.load(params_file_name, map_location='cpu'))
    predict_and_save_results(
        net=model,
        data_loader=data_loader,
        data_target_tensor=test_target_tensor,
        global_step=global_step,
        metric_method=metric_method,
        _mean=_mean,
        _std=_std,
        params_path=params_path,
        run_type=run_type,
        sw=sw,
        plot_sensor_count=10,
        forward=False
    )


if __name__ == '__main__':
    # train model
    if not only_predict:
        train_main()
    else:
        print("Only Predict")
        predict_main(
            global_step=0,
            data_loader=test_loader,
            test_target_tensor=test_target_tensor,
            metric_method=metric_method,
            _mean=_mean,
            _std=_std,
            run_type='test',
            sw=sw
        )
        print('=====Predict Success=====')
