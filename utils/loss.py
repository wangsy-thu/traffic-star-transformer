import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


def compute_val_loss(net: nn.Module, val_loader: DataLoader, criterion,
                     masked_flag: int, missing_value: float, sw: SummaryWriter,
                     epoch: int, limit=None):
    """

    :param net: 模型
    :param val_loader: Validation 加载器
    :param criterion: 损失函数
    :param masked_flag: 是否 Masked
    :param missing_value: 缺失值
    :param sw: 日志工具
    :param epoch: 训练步数
    :param limit: None
    :return: Loss Value
    """
    net.train(False)  # ensure dropout layers are in evaluation mode
    with torch.no_grad():

        tmp = []  # 记录了所有batch的loss
        prediction = []
        target = []

        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data
            outputs = net(encoder_inputs)
            prediction.append(outputs.detach().cpu().numpy())
            target.append(labels.detach().cpu().numpy())
            if masked_flag:
                loss = criterion(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)

            tmp.append(loss.item())
            if (limit is not None) and batch_index >= limit:
                break
        prediction = np.concatenate(prediction, axis=0)
        target = np.concatenate(target, axis=0)

        if epoch % 5 == 0:
            for i in range(10):
                fig = plt.figure()
                plt.title('Node {} Validation Result'.format(i))
                plot_length = int(0.2 * len(prediction))
                y_predict = prediction[:plot_length, i, -1]
                y_label = target[:plot_length, i, -1]
                plt.plot(range(plot_length), y_label, label='real')
                plt.plot(range(plot_length), y_predict, label='predict')
                plt.xlabel('time step')
                plt.ylabel('flow value')
                plt.legend()
                sw.add_figure(
                    'Node {} Validation Result Figure'.format(i),
                    fig,
                    global_step=epoch,
                    close=True
                )

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss
