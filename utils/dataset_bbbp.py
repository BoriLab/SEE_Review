import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch.nn.functional as F
import scipy.sparse as sp



# --- DataLoader에서 사용되는 collate 함수 ---
def collate_batch(batch):
    node_features, adj_matrices, edge_indices, edge_attrs, targets = zip(*batch)
    
    # 각 샘플의 최대 노드 수와 feature dimension 계산
    max_nodes = max(nf.size(0) for nf in node_features)
    max_feats = max(nf.size(1) for nf in node_features)
    
    # 각 샘플별 노드 마스크 생성
    node_masks = [
        torch.cat([torch.ones(nf.size(0)), torch.zeros(max_nodes - nf.size(0))])
        for nf in node_features
    ]
    
    # 노드 특징 패딩: (노드 수, feature dimension)
    padded_node_features = [
        F.pad(nf, (0, max_feats - nf.size(1), 0, max_nodes - nf.size(0)))
        for nf in node_features
    ]
    
    # 인접 행렬 패딩
    padded_adj_matrices = [
        F.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(1)))
        for adj in adj_matrices
    ]
    
    # 엣지 인덱스 및 엣지 속성 패딩
    padded_edge_indices = []
    padded_edge_attrs = []
    max_edges = max(ei.size(1) for ei in edge_indices)
    for ei, ea in zip(edge_indices, edge_attrs):
        pad_size = max_edges - ei.size(1)
        if pad_size > 0:
            pad_ei = torch.full((2, pad_size), -1, dtype=torch.long)  # 패딩 값은 -1 (필요 시 수정)
            ei = torch.cat([ei, pad_ei], dim=1)
            pad_ea = torch.zeros((pad_size, ea.size(1)))
            ea = torch.cat([ea, pad_ea], dim=0)
        padded_edge_indices.append(ei)
        padded_edge_attrs.append(ea)
    
    # 텐서 스택
    node_features = torch.stack(padded_node_features)
    adj_matrices = torch.stack(padded_adj_matrices)
    edge_indices = torch.stack(padded_edge_indices)
    edge_attrs = torch.stack(padded_edge_attrs)
    node_masks = torch.stack(node_masks).bool()
    targets = torch.stack(targets)
    
    return (node_features, adj_matrices, edge_indices, edge_attrs, node_masks, targets)

# --- 데이터셋 클래스 ---
class MyDataset(Dataset):
    def __init__(
        self,
        node_features,
        adj_matrices,
        edge_indices,
        edge_attrs,
        targets,
        evaluation_size=0.025,
        test_size=0.025,
        batch_size=32,
        seed=42  # seed 인자 추가
    ):
        self.node_features = node_features
        self.adj_matrices = adj_matrices
        self.edge_indices = edge_indices
        self.edge_attrs = edge_attrs
        self.targets = targets
        self.batch_size = batch_size
        self.seed = seed

        # sparse matrix 처리 및 텐서 변환
        for i in range(len(self.node_features)):
            if sp.issparse(self.node_features[i]):
                self.node_features[i] = self.node_features[i].toarray()
            self.node_features[i] = torch.tensor(self.node_features[i], dtype=torch.float)
            
            if sp.issparse(self.adj_matrices[i]):
                self.adj_matrices[i] = self.adj_matrices[i].toarray()
            self.adj_matrices[i] = torch.tensor(self.adj_matrices[i], dtype=torch.float)
            
            self.edge_indices[i] = torch.tensor(self.edge_indices[i], dtype=torch.long)
            self.edge_attrs[i] = torch.tensor(self.edge_attrs[i], dtype=torch.float)
            self.targets[i] = torch.tensor(self.targets[i], dtype=torch.float)

        # 데이터셋 분할을 위한 인덱스 셔플 (seed 고정)
        random.seed(self.seed)
        indices = list(range(len(self.node_features)))
        random.shuffle(indices)
        if evaluation_size < 1:
            evaluation_size = int(evaluation_size * len(indices))
        if test_size < 1:
            test_size = int(test_size * len(indices))
        self.indices = {
            "train": indices[test_size + evaluation_size :],
            "eval": indices[:evaluation_size],
            "test": indices[evaluation_size : test_size + evaluation_size],
        }

        self.node_feat_size = self.node_features[0].shape[1]
        self.prediction_size = self.targets[0].shape[0] if self.targets[0].dim() > 0 else 1

    def float(self):
        for i in range(len(self.node_features)):
            self.node_features[i] = self.node_features[i].float()
            self.adj_matrices[i] = self.adj_matrices[i].float()
            self.edge_attrs[i] = self.edge_attrs[i].float()
            self.targets[i] = self.targets[i].float()

    def unsqueeze_target(self):
        for i in range(len(self.targets)):
            if self.targets[i].dim() == 0:
                self.targets[i] = self.targets[i].unsqueeze(-1)

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        return (
            self.node_features[idx],
            self.adj_matrices[idx],
            self.edge_indices[idx],
            self.edge_attrs[idx],
            self.targets[idx],
        )

    def get_dataloader(self, split="train"):
        # num_workers를 사용하지 않는 경우 기본적으로 main 프로세스에서 실행되므로 별도의 worker_init_fn은 필요없음.
        sampler = SubsetRandomSampler(self.indices[split])
        return DataLoader(
            self,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_batch,
            shuffle=False  # sampler에서 이미 셔플함
        )

    def train(self):
        return self.get_dataloader(split="train")

    def eval(self):
        return self.get_dataloader(split="eval")

    def test(self):
        return self.get_dataloader(split="test")
