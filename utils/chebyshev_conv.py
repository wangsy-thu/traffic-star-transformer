import numpy as np
from scipy.sparse.linalg import eigs


def make_chebyshev_polynomial(L_tilde, K):
    """
    计算 K 阶 Chebyshev 多项式
    切比雪夫卷积核从本质上讲就是一个矩阵，1次左乘 H 就是对邻居节点抽取一次特征
    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N) 归一化的 Laplacian 矩阵
                这里可以理解为将图的邻接矩阵的一种变换
    K: the maximum order of chebyshev polynomials
                这里可以理解为抽取图结构的 K 跳邻居特征
    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    """

    # 节点数量
    N = L_tilde.shape[0]

    # 初始状态的
    chebyshev_polynomials = [np.identity(N), L_tilde.copy()]

    # 递归定义的切比雪夫多项式
    for i in range(2, K):
        chebyshev_polynomials.append(2 * L_tilde * chebyshev_polynomials[i - 1] - chebyshev_polynomials[i - 2])

    return chebyshev_polynomials


def scaled_Laplacian(W):
    """
    计算 归一化的 Laplace 矩阵
    Parameters
    ----------
    W: np.ndarray, shape (N, N), N为
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    """

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    """
    计算邻接矩阵
    :param id_filename: id 文件名
    :param distance_df_filename: 距离文件名
    :param num_of_vertices: 节点数量
    :return: A: np.ndarray, adjacency matrix
    """
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        import csv
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distanceA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)
        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distanceA[id_dict[i], id_dict[j]] = distance
            return A, distanceA

        else:
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 2:
                        i, j = int(row[0]), int(row[1])
                        A[i, j] = 1
                    elif len(row) == 3:
                        i, j, distance = int(row[0]), int(row[1]), float(row[2])
                        distanceA[i, j] = distance
                        A[i, j] = 1
                    else:
                        continue
            return A, distanceA
