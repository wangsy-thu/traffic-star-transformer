import argparse
import configparser
import os.path

import numpy as np
from tqdm import trange


def search_window_range(sequence_length: int, num_of_depend: int,
                        label_start_idx: int, predict_steps: int,
                        units: int, points_per_hour: int):
    """
    序列窗口数据区间左右边界搜索
    :param sequence_length: 序列长度
    :param num_of_depend: 需要预测的周期数
    :param label_start_idx: 标签开始索引
    :param predict_steps: 预测时间步数量
    :param units: 基本单元，单位为 hour
    :param points_per_hour: 每小时的采样点数量
    :return: 索引边界列表
    """

    # 判断参数合法性
    if points_per_hour < 0:
        raise ValueError("param points_per_hour should be greater than 0!")

    # 判断右边界是否越界，若越界，则本次搜索作废
    if label_start_idx + predict_steps > sequence_length:
        return None

    window_idx = []
    for i in range(1, num_of_depend + 1):
        window_start_idx = label_start_idx - points_per_hour * units * i
        window_end_idx = window_start_idx + predict_steps

        # 判断左边界是否越界，若越界，则本次搜索非法
        if window_start_idx > 0:
            window_idx.append((window_start_idx, window_end_idx))
        else:
            return None

    if len(window_idx) != num_of_depend:
        return None

    # 倒序取出
    return window_idx[::-1]


def get_sample(data_sequence: np.ndarray, num_of_weeks: int, num_of_days: int,
               num_of_hours: int, label_start_idx: int, predict_step: int,
               points_per_hour=12):
    """
    获取数据样本
    :param data_sequence: 数据序列
    :param num_of_weeks: 预测输入星期数
    :param num_of_days: 预测输入天数
    :param num_of_hours: 预测输入小时数
    :param label_start_idx: 标签开始索引
    :param predict_step: 预测时间步
    :param points_per_hour: 每小时包含多少采样点
    :return: (week_sample, day_sample, hour_sample, target)
                week_sample: np.ndarray (W x 12, vertices_num, features_num)
                day_sample: np.ndarray (D x 12, vertices_num, features_num)
                hour_sample: np.ndarray (H x 12, vertices_num, features_num)
    """
    week_sample, day_sample, hour_sample = None, None, None
    sequence_length = data_sequence.shape[0]

    # 判断数据是否越界
    if label_start_idx + predict_step > sequence_length:
        return week_sample, day_sample, hour_sample, None

    # 处理 Week 周期数据
    if num_of_weeks > 0:
        # 获取星期采样数据的时间窗口边界索引
        week_windows_range = search_window_range(
            sequence_length=sequence_length,
            num_of_depend=num_of_weeks,
            label_start_idx=label_start_idx,
            predict_steps=predict_step,
            units=7 * 24,
            points_per_hour=points_per_hour
        )
        # 星期窗口索引边界为空，则本次星期采样不合法
        if not week_windows_range:
            return None, None, None
        # 采集星期样本数据
        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_windows_range], axis=0)

    # 处理 day 采样数据
    if num_of_days > 0:
        # 获取日采样数据的时间窗口边界索引
        day_window_range = search_window_range(
            sequence_length=sequence_length,
            num_of_depend=num_of_days,
            label_start_idx=label_start_idx,
            predict_steps=predict_step,
            units=24,
            points_per_hour=12
        )
        # 日窗口索引边界为空，则本次日采样不合法
        if not day_window_range:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_window_range], axis=0)

    # 处理 hour 采样数据
    if num_of_hours > 0:
        # 获取小时采样数据的时间窗口边界索引
        hour_window_range = search_window_range(
            sequence_length=sequence_length,
            num_of_depend=num_of_hours,
            label_start_idx=label_start_idx,
            predict_steps=predict_step,
            units=1,
            points_per_hour=points_per_hour
        )

        # 小时窗口索引边界为空，则本次小时采样不合法
        if not hour_window_range:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_window_range], axis=0)

    # 处理标签
    target = data_sequence[label_start_idx: label_start_idx + predict_step]

    return week_sample, day_sample, hour_sample, target


def normalization(train_data: np.ndarray, val_data: np.ndarray, test_data: np.ndarray):
    """
    归一化
    :param train_data: 训练数据
    :param val_data: 验证数据
    :param test_data: 测试数据
    :return: (stats, train_norm, val_norm, test_norm)
                stats: (mean, std)
                train_norm, val_norm, test_norm: np.ndarray(S, N, F, T)
    """
    # 确保数据形状相同
    assert train_data.shape[1:] == val_data.shape[1:] and \
           val_data.shape[1:] == test_data.shape[1:]

    mean = train_data.mean(axis=(0, 1, 3), keepdims=True)
    std = train_data.std(axis=(0, 1, 3), keepdims=True)

    print('mean.shape:', mean.shape)
    print('std.shape:', std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train_data)
    val_norm = normalize(val_data)
    test_norm = normalize(test_data)

    return {"_mean": mean, "_std": std}, train_norm, val_norm, test_norm


def read_save_dataset(flow_matrix_filename: str, num_of_weeks: int, num_of_days: int,
                      num_of_hours: int, predict_step: int, points_per_hour: int, save_file=False):
    """
    读取并保存数据集，按照时间窗口将数据处理成目标格式
    :param flow_matrix_filename: 流量数据文件路径名称
    :param num_of_weeks: 星期采样
    :param num_of_days: 日采样
    :param num_of_hours: 小时采样
    :param predict_step: 预测时间步
    :param points_per_hour: 每小时时间步数量
    :param save_file: 是否保存数据文件
    :return: (feature, target)
                feature: np.ndarray(sample_num, num_of_depend x points_per_hour,
                                    vertices_num, features_num)
                target: np.ndarray(sample_num, vertices_num, feature_num)
    """
    # 读取数据文件
    data_seq: np.ndarray = np.load(flow_matrix_filename)['data']
    sequence_length = data_seq.shape[0]
    all_samples = []

    # 按照滑动窗口读取预测数据
    for idx in trange(sequence_length):
        sample = get_sample(
            data_sequence=data_seq,
            num_of_weeks=num_of_weeks,
            num_of_days=num_of_days,
            num_of_hours=num_of_hours,
            label_start_idx=idx,
            predict_step=predict_step,
            points_per_hour=points_per_hour
        )

        # 若所有采样均非法，跳过本次采样
        if (sample[0] is None) and (sample[1] is None) and (sample[2] is None):
            continue

        week_sample, day_sample, hour_sample, target = sample
        # 重整 sample 列表
        sample = []

        # 处理周采样数据
        if num_of_weeks > 0:
            # 将时间维度移到最后
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1, N, F, T)
            sample.append(week_sample)

        # 处理日采样数据
        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1, N, F, T)
            sample.append(day_sample)

        # 处理时采样数据
        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1, N, F, T)
            sample.append(hour_sample)

        # 处理目标数据集
        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1, N, T)
        sample.append(target)

        # 处理时间标签
        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1, 1)
        sample.append(time_sample)

        # [(week_sample), (day_sample), (hour_sample), target, time_sample]
        all_samples.append(sample)

    # 划分训练集与测试集
    print('=====Split Sequence Data=====')
    train_line = int(len(all_samples) * 0.6)
    test_line = int(len(all_samples) * 0.8)

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:train_line])]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[train_line: test_line])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[test_line:])]

    # 合并所有采样数据
    train_x = np.concatenate(training_set[:-2], axis=-1)
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    # 获取target集
    train_target = training_set[-2]
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    # 获取每次采样的开始时间步
    train_timestamp = training_set[-1]
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]

    stat, train_x_norm, val_x_norm, test_x_norm = normalization(
        train_x, val_x, test_x
    )

    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'timestamp': train_timestamp
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'timestamp': val_timestamp
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'timestamp': test_timestamp
        },
        'stats': {
            '_mean': stat['_mean'],
            '_std': stat['_std']
        }
    }

    # 保存训练数据到压缩文件
    if save_file:
        print('=====Saving File=====')
        file_base_name = os.path.basename(flow_matrix_filename).split('.')[0]
        file_dir_name = os.path.dirname(flow_matrix_filename)

        file_name = os.path.join(file_dir_name,
                                 file_base_name + '_r' + str(num_of_hours) +
                                 '_d' + str(num_of_days) +
                                 '_w' + str(num_of_weeks))
        print('File Name: {}'.format(file_name))
        np.savez_compressed(
            file_name,
            train_x=all_data['train']['x'],
            train_target=all_data['train']['target'],
            train_timestamp=all_data['train']['timestamp'],
            val_x=all_data['val']['x'],
            val_target=all_data['val']['target'],
            val_timestamp=all_data['val']['timestamp'],
            test_x=all_data['test']['x'],
            test_target=all_data['test']['target'],
            test_timestamp=all_data['test']['timestamp'],
            mean=all_data['stats']['_mean'],
            std=all_data['stats']['_std']
        )

    return all_data


if __name__ == '__main__':
    # prepare dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='config/PEMS04_star.conf', type=str,
                        help="configuration file path")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    print('Read configuration file: %s' % args.config)
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']

    # 获取配置
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
    num_of_weeks = int(training_config['num_of_weeks'])
    num_of_days = int(training_config['num_of_days'])
    num_of_hours = int(training_config['num_of_hours'])
    model_name = training_config['model_name']
    graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
    data = np.load(graph_signal_matrix_filename)

    all_data = read_save_dataset(
        flow_matrix_filename=graph_signal_matrix_filename,
        num_of_weeks=num_of_weeks,
        num_of_days=num_of_days,
        num_of_hours=num_of_hours,
        predict_step=num_for_predict,
        points_per_hour=points_per_hour,
        save_file=True
    )

    print('=====Validation Preprocess Result=====')
    file_data = np.load('./data/{}/{}_r{}_d{}_w{}.npz'.format(
        dataset_name, dataset_name, num_of_hours,
        num_of_days, num_of_weeks
    ))
    print('train_x shape:{}'.format(file_data['train_x'].shape))
    print('train_target shape:{}'.format(file_data['train_target'].shape))
    print('train_timestamp shape:{}'.format(file_data['train_timestamp'].shape))
    print('val_x shape:{}'.format(file_data['val_x'].shape))
    print('val_target shape:{}'.format(file_data['val_target'].shape))
    print('val_timestamp shape:{}'.format(file_data['val_timestamp'].shape))
    print('test_x shape:{}'.format(file_data['test_x'].shape))
    print('test_target shape:{}'.format(file_data['test_target'].shape))
    print('test_timestamp shape:{}'.format(file_data['test_timestamp'].shape))
    print('train std: {}'.format(file_data['std']))
    print('train mean: {}'.format(file_data['mean']))
