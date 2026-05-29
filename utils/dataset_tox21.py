import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import random
import scipy.sparse as sp

def collate_batch(batch):
    node_features, adj_matrices, edge_indices, edge_attrs, targets = zip(*batch)
    
    # 각 샘플의 노드 개수 및 feature dimension 확인
    max_nodes = max(nf.size(0) for nf in node_features)
    max_feats = max(nf.size(1) for nf in node_features)
    
    # 노드 마스크 생성 (각 샘플별 유효 노드 위치)
    node_masks = [
        torch.cat([torch.ones(nf.size(0)), torch.zeros(max_nodes - nf.size(0))])
        for nf in node_features
    ]
    
    # 노드 특징 패딩: (노드 개수, feature dimension) 모두 max 값으로 패딩
    padded_node_features = [
        F.pad(nf, (0, max_feats - nf.size(1), 0, max_nodes - nf.size(0)))
        for nf in node_features
    ]
    
    # 인접 행렬 패딩
    padded_adj_matrices = [
        F.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(1)))
        for adj in adj_matrices
    ]
    
    # edge_indices 보정을 위한 함수: 항상 [2, num_edges] 형태로 반환
    def fix_edge_index(ei):
        if ei is None or ei.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long)
        if ei.dim() == 1:
            ei = ei.unsqueeze(0)
        if ei.size(0) != 2:
            # 만약 1이면 복제
            if ei.size(0) == 1:
                ei = torch.cat([ei, ei], dim=0)
            else:
                raise ValueError("Unexpected edge_index shape: " + str(ei.shape))
        return ei
    edge_indices = [fix_edge_index(ei) for ei in edge_indices]
    
    # edge_attrs 보정을 위한 함수: 항상 2차원 텐서로 반환
    def fix_edge_attr(ea):
        if ea is None or ea.numel() == 0:
            return torch.empty((0, 0), dtype=torch.float)
        if ea.dim() < 2:
            return ea.unsqueeze(1)
        return ea
    edge_attrs = [fix_edge_attr(ea) for ea in edge_attrs]
    
    # 먼저, 배치 내 모든 non-empty edge_attr의 feature dimension의 최댓값을 구함.
    global_feat_dim = 0
    for ea in edge_attrs:
        if ea.numel() > 0:
            global_feat_dim = max(global_feat_dim, ea.size(1))
    # 만약 모두 빈 경우 global_feat_dim은 0.
    
    # 엣지 인덱스 및 엣지 속성 패딩
    padded_edge_indices = []
    padded_edge_attrs = []
    max_edges = max(ei.size(1) for ei in edge_indices) if len(edge_indices) > 0 else 0
    for ei, ea in zip(edge_indices, edge_attrs):
        # 만약 ea가 빈 텐서이면, 강제로 feature dimension을 global_feat_dim으로 설정
        if ea.numel() == 0:
            ea = torch.empty((0, global_feat_dim), dtype=torch.float)
        pad_size = max_edges - ei.size(1)
        if pad_size > 0:
            pad_ei = torch.full((2, pad_size), -1, dtype=torch.long)  # 패딩 값 -1
            ei = torch.cat([ei, pad_ei], dim=1)
            pad_ea = torch.zeros((pad_size, ea.size(1)))
            ea = torch.cat([ea, pad_ea], dim=0)
        padded_edge_indices.append(ei)
        padded_edge_attrs.append(ea)
    
    # 텐서 스택
    node_features = torch.stack(padded_node_features)
    adj_matrices = torch.stack(padded_adj_matrices)
    edge_indices = torch.stack(padded_edge_indices)
    edge_attrs = torch.stack(padded_edge_attrs) if len(padded_edge_attrs) > 0 else torch.empty((len(padded_edge_indices), 0, global_feat_dim))
    node_masks = torch.stack(node_masks).bool()
    targets = torch.stack(targets)
    
    return (node_features, adj_matrices, edge_indices, edge_attrs, node_masks, targets)

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
        # 데이터 저장
        self.node_features = node_features
        self.adj_matrices = adj_matrices
        self.edge_indices = edge_indices
        self.edge_attrs = edge_attrs
        self.targets = targets
        self.batch_size = batch_size
        self.seed = seed

        # 데이터 전처리: sparse matrix 처리 및 tensor 변환
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
            #self.targets[i] = self.targets[i].clone().detach().to(torch.float) 나중에 이걸로 ㅂ코드 변경
        # 데이터셋 분할을 위해 seed 고정 및 인덱스 셔플
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
        # 모든 split에 대해 SubsetRandomSampler 사용하여 인덱스 셔플된 데이터를 로드
        sampler = SubsetRandomSampler(self.indices[split])
        return DataLoader(
            self,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_batch,
            shuffle=False,  # sampler에서 섞으므로 DataLoader에서는 shuffle=False
        )

    def train(self):
        return self.get_dataloader(split="train")

    def eval(self):
        return self.get_dataloader(split="eval")

    def test(self):
        return self.get_dataloader(split="test")
