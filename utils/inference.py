import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from utils.common import re_normalization
from utils.metrics import masked_mae_test, masked_rmse_test, masked_mape_np


def predict_and_save_results(net: nn.Module, data_loader: DataLoader,
                             data_target_tensor: torch.Tensor, global_step: int,
                             metric_method: str, _mean, _std, params_path: str,
                             type: str, sw: SummaryWriter, plot_sensor_count: int):
    """
    :param plot_sensor_count: 绘制预测结果的传感器数量
    :param sw: 日志工具
    :param type:
    :param metric_method: 评价方法
    :param global_step: 全局训练步，这里指训练到第几步
    :param net: nn.Module
    :param data_loader: torch.model_utils.data.model_utils.DataLoader
    :param data_target_tensor: tensor
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    """
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():
        data_target_tensor = data_target_tensor.cpu().numpy()
        loader_length = len(data_loader)  # nb of batch
        prediction = []  # 存储所有batch的output
        input_mat = []  # 存储所有batch的input
        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, labels = batch_data

            input_mat.append(encoder_inputs[:, :, 0:1].cpu().numpy())  # (batch, T', 1)

            outputs = net(encoder_inputs)

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        input_mat = np.concatenate(input_mat, 0)

        input_mat = re_normalization(input_mat, _mean, _std)

        prediction = np.concatenate(prediction, 0)  # (batch, T, 1)

        # Plot Prediction Result
        for i in range(plot_sensor_count):
            fig = plt.figure()
            plt.title('Node {} Prediction Result'.format(i))
            plot_length = int(0.2 * len(prediction))
            y_predict = prediction[:plot_length, i, -1]
            y_label = data_target_tensor[:plot_length, i, -1]
            plt.plot(range(plot_length), y_label, label='real')
            plt.plot(range(plot_length), y_predict, label='predict')
            plt.xlabel('time step')
            plt.ylabel('flow value')
            plt.legend()
            sw.add_figure(
                'Node {} Predict Result Figure'.format(i),
                fig,
                global_step=global_step,
                close=True
            )

        print('input:', input_mat.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, input=input_mat, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (global_step, i))
            if metric_method == 'mask':
                mae = masked_mae_test(data_target_tensor[:, :, i], prediction[:, :, i], 0.0)
                rmse = masked_rmse_test(data_target_tensor[:, :, i], prediction[:, :, i], 0.0)
                mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)
            else:
                mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i])
                rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
                mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)
            excel_list.extend([mae, rmse, mape])

        # print overall results
        if metric_method == 'mask':
            mae = masked_mae_test(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
            rmse = masked_rmse_test(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
            mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        else:
            mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
            rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
            mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % mae)
        print('all RMSE: %.2f' % rmse)
        print('all MAPE: %.2f' % mape)
        excel_list.extend([mae, rmse, mape])
        print(excel_list)
