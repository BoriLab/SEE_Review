
"""
MoleculeNet → Pickle 변환 스크립트
 - one_hot_nodes : float32  (원본 그대로)
 - adjacency     : sparse CSR, float32
 - edge_index    : int16
 - edge_attr     : float16
 - 압축          : 사용하지 않음 (pickle protocol 5)
"""

import os
import random
import pickle
import numpy as np
import torch
# import pandas as pd
from torch_geometric.datasets import MoleculeNet
from rdkit import Chem
from rdkit.Chem import AllChem
import scipy.sparse as sp
from scipy.linalg import eigh

# --------------------------------------------------------------------------- #
# 시드 고정
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed);  np.random.seed(seed);  torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True)

# --------------------------------------------------------------------------- #
# 2D 좌표
# --------------------------------------------------------------------------- #

def get_2d_coordinates(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    return np.array([[conf.GetAtomPosition(i).x,
                      conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())],
                    dtype=np.float32)

# --------------------------------------------------------------------------- #
# Spectral Edge Encoding
# --------------------------------------------------------------------------- #

def spectral_edge_encoding(coords: np.ndarray, k_eigen=8, gamma=0.5):
    n = coords.shape[0]
    A = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(coords[i] - coords[j]) < 2.5:
                A[i, j] = A[j, i] = 1
    D, L = np.diag(A.sum(1)), None
    L = D - A
    k_eigen = min(k_eigen, n - 1) if n > 1 else 1
    eigvals, eigvecs = np.linalg.eigh(L)
    eigvals, eigvecs = eigvals[:k_eigen], eigvecs[:, :k_eigen]

    edge_index, edge_attr = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                Δ = np.zeros((n, n));  Δ[i, i] = Δ[j, j] = 1;  Δ[i, j] = Δ[j, i] = -1
                δλ = np.array([eigvecs[:, k].T @ Δ @ eigvecs[:, k] for k in range(k_eigen)],
                              dtype=np.float32)
                ess = np.sum(np.exp(-gamma * eigvals) * np.abs(δλ))
                edge_index += [[i, j], [j, i]]
                edge_attr  += [[ess],  [ess]]
    return (np.asarray(edge_index, dtype=np.int16).T,
            np.asarray(edge_attr,  dtype=np.float16))

# --------------------------------------------------------------------------- #
# sparse adjacency (CSR, float32)
# --------------------------------------------------------------------------- #

def make_sparse_adjacency(coords: np.ndarray, thresh=None):
    n = coords.shape[0];  row, col, dist = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if (thresh is None) or (d < thresh):
                row += [i, j];  col += [j, i];  dist += [d, d]
    return sp.csr_matrix((np.asarray(dist, dtype=np.float32),
                          (row, col)),
                         shape=(n, n),
                         dtype=np.float32)

# --------------------------------------------------------------------------- #
# 메인 전처리
# --------------------------------------------------------------------------- #

def load_and_save_moleculenet_dataset(dataset_name: str,
                                      root: str,
                                      pkl_path: str,
                                      seed: int = 42):
    set_seed(seed)
    ds = MoleculeNet(root=root, name=dataset_name)

    data_list = []
    for data in ds:
        # one_hot_nodes: 원본 float32 유지
        one_hot = data.x.cpu().numpy()

        # target: Tensor → float32 numpy
        target = data.y.to(torch.float32).cpu().numpy()

        sample = dict(one_hot_nodes=one_hot,
                      edge_index_2d=None,
                      edge_attr_2d=None,
                      smiles=data.smiles if hasattr(data, 'smiles') else None,
                      target=target)

        smiles = getattr(data, "smiles", None)
        if smiles:
            coords = get_2d_coordinates(smiles)
            if coords is not None:
                sample["coords"] = coords
                sample["adj"]    = make_sparse_adjacency(coords)
                ei, ea = spectral_edge_encoding(coords, k_eigen=5, gamma=0.5)
                sample["edge_index_2d"], sample["edge_attr_2d"] = ei, ea
        data_list.append(sample)

    keys = ["one_hot_nodes", "adj", "edge_index_2d",
            "edge_attr_2d", "target",'smiles']
    data_dict = {k: [s.get(k) for s in data_list] for k in keys}

    with open(pkl_path, "wb") as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data_dict

# --------------------------------------------------------------------------- #
#  실행
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    base_root = "/root/2025/dataset"
    out_dir   = "/root/2025/sse_moleculenet/data/moleculenet_dataset/data_pkl"
    os.makedirs(out_dir, exist_ok=True)

    for dset in ["MUV"]:                   # 필요 시 교체
        root = os.path.join(base_root, dset)
        pkl  = os.path.join(out_dir, f"{dset.lower()}_data.pkl")
        info = load_and_save_moleculenet_dataset(dset, root, pkl, seed=42)
        print(f"{dset}: {len(info['one_hot_nodes'])} samples → {pkl}")
