import os
import random
import pickle
import numpy as np
import torch
from torch_geometric.datasets import MoleculeNet
from rdkit import Chem
from rdkit.Chem import AllChem
import scipy.sparse as sp

##############################################################################
# 시드 고정
##############################################################################
def set_seed(seed=42):
    # Python 내장 해시 seed 고정 
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS deterministic을 위한 환경 변수 설정 
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # Python, NumPy, PyTorch의 난수 seed를 동일하게 고정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    #GPU 사용 시 CUDA 난수 seed도 함께 고정
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN의 비결정적 최적화를 비활성화
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # deterministic algorithm만 사용하도록 설정
    torch.use_deterministic_algorithms(True)

##############################################################################
# RDKit 기반 SMILES-to-2D coordinate 변환 함수
##############################################################################
def get_2d_coordinates(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    return np.array([[conf.GetAtomPosition(i).x,
                      conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())],
                    dtype=np.float32)

##############################################################################
# Spectral Edge Encoding 계산 함수
##############################################################################
def spectral_edge_encoding(coords: np.ndarray, k_eigen=5, gamma=0.5):
    """
    2D 좌표 기반 graph에서 edge별 Spectral Edge Encoding 값을 계산

    핵심 흐름
    1. 2D 좌표 거리 threshold를 기준으로 binary adjacency A를 생성
    2. combinatorial Laplacian L = D - A를 구성
    3. Laplacian eigendecomposition을 통해 low-frequency eigenmode를 얻음
    4. 각 edge를 Laplacian perturbation으로 보고, 해당 edge가 low-frequency
       eigenvalue에 주는 변화량을 ESS score로 계산
    5. PyTorch Geometric의 directed COO 형식에 맞춰 edge를 양방향으로 저장
    """
    n = coords.shape[0]

    # SEE 계산을 위한 binary adjacency
    A = np.zeros((n, n), dtype=np.int8)

    # 2D coordinate 상에서 일정 거리 이내의 atom pair를 edge로 연결
    # 현재 threshold=2.5는 실험적으로 설정한 hyperparameter
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(coords[i] - coords[j]) < 2.5:
                A[i, j] = A[j, i] = 1
    # 분자 그래프를 undirected relation으로 구성했기 때문에 A는 대칭이며,
    # 이에 따라 L도 실수 대칭 행렬이 되어 np.linalg.eigh를 사용할 수 있음.            
    D = np.diag(A.sum(1))
    L = D - A

    # 전체 spectrum이 아니라 low-frequency mode만 사용
    # k_eigen이 node 수보다 커지는 경우를 방지
    k_eigen = min(k_eigen, n - 1) if n > 1 else 1
    eigvals, eigvecs = np.linalg.eigh(L) # 고유분해
    eigvals, eigvecs = eigvals[:k_eigen], eigvecs[:, :k_eigen]

    edge_index, edge_attr = [], []
    
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                # Edge (i, j)를 제거하는 perturbation은 ΔL_ij = (e_i - e_j)(e_i - e_j)^T 형태로 표현
                # 현재 구현은 수식과 대응관계를 명확히 보이기 위해 ΔL matrix를 직접 구성
                delta_L = np.zeros((n, n))
                delta_L[i, i] = delta_L[j, j] = 1
                delta_L[i, j] = delta_L[j, i] = -1
                
                # delta_L : edge 하나가 Laplacian에 영향을 주는지 perturbation matrix
                # Rayleigh quotient 기반 1차 근사:
                # Δλ_k(e_ij) ≈ u_k^T ΔL_ij u_k
                delta_lambda = np.array(
                    [eigvecs[:, k].T @ delta_L @ eigvecs[:, k]
                     for k in range(k_eigen)],
                    dtype=np.float32,
                )

                # Low-frequency eigenvalue에 더 큰 가중치를 주기 위해 exp(-gamma * lambda)를 곱
                # ESS가 클수록 해당 edge가 graph의 전역 구조에 더 큰 영향을 주는 것으로 해석
                ess = np.sum(np.exp(-gamma * eigvals) * np.abs(delta_lambda))

                # 분자 그래프의 원자 간 관계는 방향성이 없는 undirected relation
                # PyG의 edge_index는 directed COO 형식이므로, undirected edge 하나를
                # (i, j), (j, i) 두 방향으로 저장
                edge_index += [[i, j], [j, i]]
                edge_attr += [[ess], [ess]]

    # 저장 공간 절약을 위해 edge_index는 int16, edge_attr은 float32으로 저장
    # 학습 단계에서 PyG Data 객체로 변환할 때 edge_index는 torch.long으로 cast
    return (
        np.asarray(edge_index, dtype=np.int16).T,
        np.asarray(edge_attr, dtype=np.float32),
    )

##############################################################################
# 2D 좌표 기반 sparse 인접 행렬 생성 함수 -> 저장 공간 측면에서 개선
##############################################################################
def make_sparse_adjacency(coords: np.ndarray, thresh=None):
    n = coords.shape[0];  row, col, dist = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if (thresh is None) or (d < thresh):
                # 분자 그래프를 undirected relation으로 보기 때문에 거리 정보도 양방향으로 저장
                row += [i, j];  col += [j, i];  dist += [d, d]
    return sp.csr_matrix((np.asarray(dist, dtype=np.float32),
                          (row, col)),
                         shape=(n, n),
                         dtype=np.float32)

##############################################################################
# MoleculeNet 데이터셋 전처리 및 저장 함수
##############################################################################
def load_and_save_moleculenet_dataset(dataset_name: str,
                                      root: str,
                                      pkl_path: str,
                                      seed: int = 42):
    """
    MoleculeNet dataset을 불러와 학습에 필요한 graph/geometry 정보를 pickle로 저장

    저장 목적
    - RDKit 2D coordinate 생성
    - sparse pairwise distance matrix 생성
    - SEE edge_index / edge_attr 사전 계산
    - target label 저장

    SEE는 eigendecomposition 비용이 있으므로 학습 중 매번 계산하지 않고, preprocessing 단계에서 offline으로 계산해 저장
    """
    set_seed(seed)
    # 데이터셋 불러오기:
    ds = MoleculeNet(root=root, name=dataset_name)

    data_list = []
    for data in ds:
        # PyG에서 제공하는 atom feature matrix를 그대로 numpy로 변환
        one_hot = data.x.cpu().numpy()

        # MoleculeNet label을 float32 numpy array로 변환
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
                sample["adj"] = make_sparse_adjacency(coords)
                ei, ea = spectral_edge_encoding(coords, k_eigen=5, gamma=0.5)
                sample["edge_index_2d"], sample["edge_attr_2d"] = ei, ea

        data_list.append(sample)

    # 학습 단계에서 사용할 핵심 feature만 선택하여 dictionary 형태로 저장
    keys = ["one_hot_nodes", 
            "adj", 
            "edge_index_2d",
            "edge_attr_2d", 
            "target",
            "smiles"
        ]
    data_dict = {key: [sample.get(key) for sample in data_list] for key in keys}

    # 결과를 pickle 파일로 저장
    with open(pkl_path, "wb") as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data_dict

##############################################################################
#  실행
##############################################################################

if __name__ == "__main__":
    base_root = "/root/2025/dataset"
    out_dir   = "/root/2025/sse_moleculenet/data/moleculenet_dataset/data_pkl"
    os.makedirs(out_dir, exist_ok=True)

    # 데이터셋 지정해서 전처리
    for dset in ["MUV"]:
        root = os.path.join(base_root, dset)
        pkl  = os.path.join(out_dir, f"{dset.lower()}_data.pkl")
        info = load_and_save_moleculenet_dataset(dset, root, pkl, seed=42)
        print(f"{dset}: {len(info['one_hot_nodes'])} samples → {pkl}")
