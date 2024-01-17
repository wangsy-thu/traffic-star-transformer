import os.path

import numpy as np
import torch
import torch.utils.data


def make_flow_data_loader(flow_matrix_filename: str, num_of_weeks: int,
                          num_of_days: int, num_of_hours: int, batch_size: int,
                          device, shuffle=True):
    """
    创建 DataLoader 对象
    :param device: 硬件设备
    :param flow_matrix_filename: 流量数据基本文件
    :param num_of_weeks: 星期采样
    :param num_of_days: 日采样
    :param num_of_hours: 小时采样
    :param batch_size: 批量数
    :param model_name: 模型名称
    :param shuffle: 是否打乱
    :return: (train_loader, val_loader, test_loader, test_target, _mean, _std)
            train_loader: 训练数据集加载器
            val_loader: 验证数据集加载器
            test_loader: 测试数据集加载器
            test_target: 测试数据标签矩阵
            _mean: 训练数据平均值
            _std: 训练数据方差
    """
    file_base_name = os.path.basename(flow_matrix_filename).split('.')[0]
    file_dir_name = os.path.dirname(flow_matrix_filename)
    file_name = os.path.join(
        file_dir_name,
        file_base_name + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)
    )
    print('=====Load File=====')
    file_data = np.load(file_name + '.npz')

    # 读取矩阵信息
    train_x = file_data['train_x']
    train_target = file_data['train_target']

    val_x = file_data['val_x']
    val_target = file_data['val_target']

    test_x = file_data['test_x']
    test_target = file_data['test_target']

    mean = file_data['mean']
    std = file_data['std']

    # Train Loader
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(device)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(device)
    train_dataset = torch.utils.data.TensorDataset(
        train_x_tensor, train_target_tensor
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )

    # Validation Loader
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(device)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(device)
    val_dataset = torch.utils.data.TensorDataset(
        val_x_tensor, val_target_tensor
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Test Loader
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(device)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(device)
    test_dataset = torch.utils.data.TensorDataset(
        test_x_tensor, test_target_tensor
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    print('=====Data Loader Created=====')
    print("trainLoader: {}".format(train_x_tensor.size()))
    print("valLoader: {}".format(val_x_tensor.size()))
    print("testLoader: {}".format(test_x_tensor.size()))

    return train_loader, val_loader, test_loader, test_target_tensor, mean, std


if __name__ == '__main__':
    train_loader, val_loader, test_loader, test_target, _mean, _std = make_flow_data_loader(
        flow_matrix_filename='../data/PEMS04/PEMS04.npz',
        num_of_weeks=0,
        num_of_days=0,
        num_of_hours=1,
        batch_size=64,
        device=torch.device('cpu'),
        shuffle=True
    )

    # Validate Loader
    for batch in train_loader:
        data_x, label = batch
        print('train shape: {}'.format(data_x.size()))
        print('label shape: {}'.format(label.size()))
        break

    # Validate Test Tensor
    print('Test Tensor Size: {}'.format(test_target.size()))
