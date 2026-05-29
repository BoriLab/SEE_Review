import os
import random
import pickle
import numpy as np
import torch
import pandas as pd
from torch_geometric.datasets import MoleculeNet
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.linalg import eigh

def get_murcko_scaffold(smiles, include_chirality=True):
    """
    주어진 SMILES 문자열로부터 Murcko Scaffold를 계산하여 canonical SMILES로 반환합니다.
    만약 변환에 실패하면 None을 반환합니다.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None:
        return None
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=include_chirality)
##############################################################################
# Tox21Dataset 클래스
##############################################################################
from utils.dataset_tox21 import MyDataset  # MyDataset를 그대로 상속받음

class Tox21Dataset(MyDataset):
    def __init__(
        self,
        path="",
        evaluation_size=0.1,
        test_size=0.1,
        batch_size=128,
        seed=42,
        scaffold_split=True  # scaffold 분할 사용 여부
    ):
        # 피클 파일로부터 전처리된 Tox21 데이터를 로드
        data = pickle.load(open(path, "rb"))
        if isinstance(data, list):
            keys = data[0].keys()
            data_dict = {key: [sample[key] for sample in data] for key in keys}
            data = data_dict
        self.data = data
        
        # MyDataset의 입력: one_hot_nodes, adj, edge_index_2d, edge_attr_2d, target
        super().__init__(
            data["one_hot_nodes"],
            data["adj"],
            data["edge_index_2d"],
            data["edge_attr_2d"],
            data["target"],
            evaluation_size=evaluation_size,
            test_size=test_size,
            batch_size=batch_size,
            seed=seed
        )
        
        self.edge_attr_dim = data["edge_attr_2d"][0].shape[1] if data["edge_attr_2d"] is not None else None
        
        if scaffold_split:
            # scaffold split: SMILES 정보를 기반으로 Murcko scaffold를 계산하여 데이터를 split
            smiles_list = data["smiles"]
            scaffold_dict = {}
            for i, smi in enumerate(smiles_list):
                murcko = get_murcko_scaffold(smi)
                key = murcko if murcko is not None else smi
                if key not in scaffold_dict:
                    scaffold_dict[key] = []
                scaffold_dict[key].append(i)
            scaffold_groups = list(scaffold_dict.values())
            scaffold_groups = sorted(scaffold_groups, key=lambda x: len(x), reverse=True)
            
            total = len(smiles_list)
            eval_count = int(evaluation_size * total)
            test_count = int(test_size * total)
            train_indices = []
            eval_indices = []
            test_indices = []
            for group in scaffold_groups:
                if len(test_indices) + len(group) <= test_count:
                    test_indices.extend(group)
                elif len(eval_indices) + len(group) <= eval_count:
                    eval_indices.extend(group)
                else:
                    train_indices.extend(group)
            
            self.indices = {
                "train": eval_indices,
                "eval": eval_indices,
                "test": test_indices,
            }
            random.seed(seed)
            random.shuffle(self.indices["train"])
            random.shuffle(self.indices["eval"])
            random.shuffle(self.indices["test"])


#   # 디버깅 출력 (옵션)
#         print("Total molecules:", total)
#         print("Train set size:", len(eval_indices))
#         print("Valid set size:", len(eval_indices))
#         print("Test set size:", len(test_indices))
