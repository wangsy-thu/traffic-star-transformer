import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset_name = "PEMS08"
    npz_data = np.load('../data/{}/{}.npz'.format(dataset_name, dataset_name))
    all_time_data = npz_data['data']
    print('Dataset {} Shape: {}'.format(dataset_name, all_time_data.shape))
    feature_num = all_time_data.shape[-1]

    # 1-plot [one node, all feature, all time] figure
    single_node_data = all_time_data[:4 * 24 * 12, 0, :]
    print(single_node_data.shape)
    for i in range(feature_num):
        plt.title('time-feature{}'.format(i + 1))
        plt.plot([k for k in range(single_node_data.shape[0])], single_node_data[:, i])
        plt.xlabel('time step')
        plt.ylabel('value')
        plt.savefig('./plot_feature/{}/time-feature{}.png'.format(dataset_name, i + 1))
        plt.figure()

    # 2-plot [all node, all feature, single time] figure
    single_time_data = all_time_data[0, :, :]
    for i in range(feature_num):
        plt.title('node-feature0')
        plt.plot([k for k in range(single_time_data.shape[0])], single_time_data[:, i])
        plt.xlabel('node id')
        plt.ylabel('value')
        plt.savefig('./plot_feature/{}/node-feature{}.png'.format(dataset_name, i + 1))
        plt.figure()

    plt.title('node-features')
    for i in range(feature_num):
        plt.plot([k for k in range(single_time_data.shape[0])], single_time_data[:, i],
                 label='feature{}'.format(i + 1))
    plt.xlabel('node id')
    plt.ylabel('value')
    plt.legend()
    plt.savefig('./plot_feature/{}/node-features.png'.format(dataset_name))
    plt.figure()
