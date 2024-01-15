import networkx as nx
import numpy as np
import pandas as pd


def get_edge_index(distance_df_filename):
    """
    获取 PyG 正向传播需要的 Edge Index
    :param distance_df_filename: 距离文件名
    :return: A: np.ndarray, adjacency matrix
    """
    edges = pd.read_csv(distance_df_filename)
    G: nx.Graph = nx.from_pandas_edgelist(edges, source='from', target='to')
    edge_index = np.zeros((2, len(G.edges)))
    for idx, e in enumerate(G.edges):
        edge_index[0, idx] = e[0]
        edge_index[1, idx] = e[1]
    return edge_index


if __name__ == '__main__':
    data_dir = '../data/PEMS04/PEMS04.csv'
    edge_index = get_edge_index(data_dir)
    print(edge_index.shape)
