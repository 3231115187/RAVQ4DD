import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import from_scipy_sparse_matrix
import os
import scipy

class MedicalNCDataset:

    def __init__(self, data_dir):
        self.name = 'medical'

        from data_loader import data_loader
        #数据路径
        dl = data_loader(r"D:\text4")

        node_features_list = []
        node_type_list = []
        num_types = len(dl.nodes['count'])

        for i in range(num_types):
            count = dl.nodes['count'][i]
            feature_matrix = dl.nodes['attr'][i]
            # 如果没有特征，用单位矩阵初始化 (One-hot)
            if feature_matrix is None:
                feature_matrix = sp.eye(count)
            # 转换为 Tensor
            if scipy.sparse.issparse(feature_matrix):
                feature_matrix = feature_matrix.todense()

            node_features_list.append(feature_matrix)
            types = torch.full((count,), i, dtype=torch.long)
            node_type_list.append(types)

        # 拼接所有节点特征
        node_feat = np.vstack(node_features_list).astype(np.float32)
        node_feat = torch.tensor(node_feat, dtype=torch.float)
        node_type = torch.cat(node_type_list, dim=0)


        all_rows = []
        all_cols = []
        all_edge_types = []

        #0: Patient-Drug, 1: Patient-Procedure, 2: Drug-Patient, 3: Procedure-Patient
        for r_id, adj in dl.links['data'].items():
            adj_coo = adj.tocoo()
            all_rows.append(adj_coo.row)
            all_cols.append(adj_coo.col)
            all_edge_types.append(np.full(adj_coo.row.shape, r_id))

        #合并所有边
        row = np.concatenate(all_rows)
        col = np.concatenate(all_cols)
        edge_type_np = np.concatenate(all_edge_types)

        edge_index = torch.tensor(np.vstack((row, col)), dtype=torch.long)
        edge_type = torch.tensor(edge_type_np, dtype=torch.long)


        num_nodes = node_feat.shape[0]
        labels = np.zeros((num_nodes, dl.labels_train['num_classes']), dtype=int)

        train_mask = dl.labels_train['mask']
        test_mask = dl.labels_test['mask']
        train_idx_arr = np.nonzero(train_mask)[0]
        test_idx_arr = np.nonzero(test_mask)[0]

        #划分验证集
        val_ratio = 0.2
        split = int(train_idx_arr.shape[0] * (1 - val_ratio))
        real_train_idx = train_idx_arr[:split]
        val_idx = train_idx_arr[split:]

        if dl.labels_train['data'] is not None:

            labels[train_mask] = dl.labels_train['data'][train_mask]

        if dl.labels_test['data'] is not None:
            labels[test_mask] = dl.labels_test['data'][test_mask]


        labels = labels.argmax(axis=1)
        labels = torch.tensor(labels, dtype=torch.long)

        self.graph = {
            'edge_index': edge_index,
            'edge_type': edge_type,
            'node_feat': node_feat,
            'node_type': node_type,
            'num_nodes': num_nodes
        }
        self.label = labels
        self.num_classes = dl.labels_train['num_classes']
        self.train_idx = torch.tensor(real_train_idx, dtype=torch.long)
        self.valid_idx = torch.tensor(val_idx, dtype=torch.long)
        self.test_idx = torch.tensor(test_idx_arr, dtype=torch.long)

    def get_idx_split(self):
        return {'train': self.train_idx, 'valid': self.valid_idx, 'test': self.test_idx}

    def __getitem__(self, idx):
        return self.graph, self.label

    def __len__(self):
        return 1


def load_medical_dataset(data_dir, name):
    return MedicalNCDataset(data_dir)