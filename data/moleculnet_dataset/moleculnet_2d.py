
import os
import random
import pickle
import numpy as np
import torch
import pandas as pd
from torch_geometric.datasets import MoleculeNet
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.linalg import eigh

##############################################################################
# 시드 고정 함수
##############################################################################
def set_seed(seed=42):
    # Python 내장 해시 seed 고정
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS 결정론성을 위한 환경 변수 설정 (PyTorch 1.8 이상 권장)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # PyTorch의 결정론적 알고리즘 사용 (PyTorch 1.8 이상)
    torch.use_deterministic_algorithms(True)
##############################################################################
# RDKit을 이용한 2D 좌표 생성 함수
##############################################################################
def get_2d_coordinates(smiles):
    """
    SMILES 문자열을 받아 RDKit을 사용하여 2D 좌표를 생성합니다.
    반환값: numpy array (num_atoms x 2) 또는 생성 실패 시 None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y])
    return np.array(coords)
##############################################################################
# Spectral Edge Encoding 함수 (2D 좌표 기반)
##############################################################################
def spectral_edge_encoding(coords, k_eigen=5, gamma=0.5): #HIV,MUV 8 0.7 나머지 5 0.5
    n = coords.shape[0]
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(coords[i] - coords[j]) < 2.5: # hyper param
                A[i, j] = A[j, i] = 1

    D = np.diag(A.sum(axis=1))
    L = D - A

    k_eigen = min(k_eigen, n - 1) if n > 1 else 1  
    eigvals, eigvecs = np.linalg.eigh(L)
    eigvals, eigvecs = eigvals[:k_eigen], eigvecs[:, :k_eigen]

    edge_index, edge_attr = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                delta_L = np.zeros((n, n))
                delta_L[i, i] = delta_L[j, j] = 1
                delta_L[i, j] = delta_L[j, i] = -1
                delta_lambda = np.array([eigvecs[:, k].T @ delta_L @ eigvecs[:, k] for k in range(k_eigen)])
                ESS = np.sum(np.exp(-gamma * eigvals) * np.abs(delta_lambda))
                edge_index += [[i, j], [j, i]]
                edge_attr += [[ESS], [ESS]]
    return np.array(edge_index).T, np.array(edge_attr), A
##############################################################################
# 2D 좌표 기반 인접 행렬 생성 함수
##############################################################################
def make_adjacency_by_distance(coor):
    n = coor.shape[0]
    adj = np.zeros((n, n))
    for j in range(n):
        for k in range(j + 1, n):
            d = np.linalg.norm(coor[j] - coor[k])
            adj[j, k] = d
            adj[k, j] = d
    return adj

##############################################################################
# MoleculeNet 데이터셋 전처리 및 저장 함수
##############################################################################
def load_and_save_moleculenet_dataset(dataset_name, root, pkl_path, seed=42):
    """
    dataset_name: 전처리할 데이터셋 이름 (예: 'BBBP', 'Tox21', 'Toxcast', ... )
    root: 데이터셋 다운로드/저장 경로
    pkl_path: 저장할 pickle 파일의 경로
    seed: 난수 고정 값
    """
    set_seed(seed)
    # 데이터셋 불러오기: 이름만 바꿔주면 여러 데이터셋 처리 가능
    dataset = MoleculeNet(root=root, name=dataset_name)
    
    data_list = []
    for data in dataset:
        # data.x는 이미 원-핫 인코딩이 된 정수형 텐서이므로, 바로 리스트로 변환하여 사용합니다.
        one_hot_nodes = data.x.tolist() if data.x is not None else None
        
        sample = {
            "one_hot_nodes": one_hot_nodes,  # 추가 전처리 없이 바로 사용
            "edge_index_2d": None,     # 기본 간선 정보
            "edge_attr_2d": None,
            "smiles": data.smiles if hasattr(data, 'smiles') else None,
            "target": data.y,                  # 이진 분류 타겟
        }
        if sample["smiles"] is not None:
            coords = get_2d_coordinates(sample["smiles"])
            if coords is not None:
                sample["coords"] = coords
                sample["adj"] = make_adjacency_by_distance(coords)
                sample["edge_index_2d"], sample["edge_attr_2d"], _ = spectral_edge_encoding(coords, k_eigen=8, gamma=0.7)
            else:
                sample["coords"] = None
                sample["adj"] = None
                sample["edge_index_2d"] = None
                sample["edge_attr_2d"] = None
        data_list.append(sample)
    
    # 필요한 키만 모아서 dict로 저장 (필요한 경우 키를 추가/변경)
    selected_keys = ["one_hot_nodes", "adj", "edge_index_2d", "edge_attr_2d", "target",'smiles']
    data_dict = {key: [sample.get(key) for sample in data_list] for key in selected_keys}
    
    with open(pkl_path, "wb") as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data_dict
    

##############################################################################
# 사용 예시: 여러 데이터셋을 한 번에 전처리
##############################################################################
if __name__ == "__main__":
    # 처리할 데이터셋 목록
    datasets = ['Clintox']
    #datasets = ['Tox21','ToxCast'] 이놈들은 난 따로 할예정
    # datasets = ['MUV','HIV','PCBA'] # 앞에와 다르게 꽤 큰 데이터셋
    # 데이터 저장 경로 예시 (각 데이터셋별 폴더 생성)
    base_root = "/root/2025/dataset"
    
    for dset in datasets:
        print(f"Processing dataset: {dset}")
        dataset_root = os.path.join(base_root, dset)
        pkl_path = f"/root/2025/sse_moleculenet/data/moleculenet_dataset/data_pkl/{dset.lower()}_data.pkl"   # 예: bbbp_data.pkl, tox21_data.pkl 등
        data_dict = load_and_save_moleculenet_dataset(dataset_name=dset, root=dataset_root, pkl_path=pkl_path, seed=42)
        print(f"{dset} -> {len(data_dict['one_hot_nodes'])} samples saved in {pkl_path}\n")

# %%
# import pickle
# import numpy as np
# # 불러올 pickle 파일 경로 (필요에 따라 경로를 수정하세요)
# pkl_file = '/root/2025/sse_moleculenet/data/moleculenet_dataset/data_pkl/hiv_data.pkl'

# # pickle 파일 열기 및 데이터 로드
# with open(pkl_file, 'rb') as f:
#     data = pickle.load(f)

# # 데이터 타입 출력
# sssss = data["one_hot_nodes"][3]
# print(sssss)
# print(len(sssss))
# print(len(data["target"]))
# print(data["edge_index_2d"][0])

# for key in data:
#     print(f"{key}: {len(data[key])} samples")
#     # 예: data_dict는 전처리 함수에서 반환된 딕셔너리라고 가정
#     targets = np.array(data["target"])  # 모든 타겟을 numpy 배열로 변환
#     has_nan = np.isnan(targets).any()
#     print("타겟에 NaN이 있는가?", has_nan)  # 이건 exp에서 처리함

# # %%


# %%
