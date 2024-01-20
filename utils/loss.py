import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models.star_transformer import StarTransformer


def compute_val_loss(net: StarTransformer, val_loader: DataLoader, criterion, device,
                     masked_flag: int, missing_value: float, sw: SummaryWriter, plot_every,
                     epoch: int, limit=None):
    """

    :param net: 模型
    :param val_loader: Validation 加载器
    :param device: 设备名称
    :param plot_every: 绘制间隔
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
            batch_size, vertices_num, _ = labels.size()
            start_engine = -torch.ones((batch_size, vertices_num, 1, 1)).to(device)
            labels = labels.unsqueeze(2)
            labels_input = torch.cat([start_engine, labels], dim=-1)
            outputs = net(encoder_inputs, labels_input)
            prediction.append(outputs[:, :, :, :-1].detach().cpu().numpy())
            target.append(labels.detach().cpu().numpy())
            if masked_flag:
                loss = criterion(outputs[:, :, :, :-1], labels, missing_value)
            else:
                loss = criterion(outputs[:, :, :, :-1], labels)

            tmp.append(loss.item())
            if (limit is not None) and batch_index >= limit:
                break
        prediction = np.concatenate(prediction, axis=0)
        target = np.concatenate(target, axis=0)

        if epoch % plot_every == 0:
            for i in range(10):
                fig = plt.figure()
                plt.title('Node {} Validation Result'.format(i))
                plot_length = int(0.2 * len(prediction))
                y_predict = prediction[:plot_length, i, 0, -1]
                y_label = target[:plot_length, i, 0, -1]
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
