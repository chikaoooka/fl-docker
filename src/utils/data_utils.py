import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


def load_mnist(root='../../data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transform)
    return dataset


def create_non_iid_data(dataset, num_clients, client_id):
    num_shards = num_clients * 2
    num_imgs = len(dataset) // num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    rand_set = set(np.random.choice(idx_shard, 2, replace=False))
    idx_shard = list(set(idx_shard) - rand_set)
    for rand in rand_set:
        dict_users[client_id] = np.concatenate(
            (dict_users[client_id], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    return dict_users[client_id]
