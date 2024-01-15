import json

import numpy as np


def byte2array(byte_data: bytes) -> np.ndarray:
    """
    将字节流转化成 numpy 的 Array
    :param byte_data: 字节流
    :return: 数组
    """
    json_data = json.loads(byte_data.decode())
    vertices_num = len(json_data)
    flow_mat = np.zeros(shape=(1, vertices_num, 12))
    for i in range(vertices_num):
        for j in range(12):
            flow_mat[0, i, j] = json_data[str(i)][j]

    return np.expand_dims(flow_mat, axis=2)


def array2byte(flow_mat: np.ndarray) -> bytes:
    """
    Numpy 中的 Array 转化为字节流
    :param flow_mat: 流量数组
    :return: 字节流
    """
    json_dict = {}
    flow_mat = flow_mat.astype(np.float64)
    _, vertices_num, step_num = flow_mat.shape

    # 初始化
    for i in range(vertices_num):
        json_dict[i] = []

    for i in range(vertices_num):
        for j in range(step_num):
            json_dict[i].append(flow_mat[0, i, j])

    return json.dumps(
        json_dict
    ).encode()
