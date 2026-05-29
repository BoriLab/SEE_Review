from utils.dataset_toxcast import MyDataset
import pickle
import random
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

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

class ToxCastDataset(MyDataset):
    def __init__(
        self,
        path="",
        evaluation_size=0.1,
        test_size=0.1,
        batch_size=128,
        seed=42,
        scaffold_split=True  # scaffold 분할 사용 여부
    ):
        # 피클 파일로부터 전처리된 ToxCast 데이터를 로드
        data = pickle.load(open(path, "rb"))
        # 만약 data가 리스트라면, key별로 모아서 딕셔너리 형태로 변환
        if isinstance(data, list):
            keys = data[0].keys()
            data_dict = {key: [sample[key] for sample in data] for key in keys}
            data = data_dict
        
        # scaffold_split을 사용하려면 SMILES 정보가 필요합니다.
        if scaffold_split and "smiles" not in data:
            print("Warning: SMILES 정보가 없어 scaffold split을 수행할 수 없습니다. 랜덤 split을 사용합니다.")
            scaffold_split = False
        
        self.data = data  # scaffold 분할에 사용하기 위해 전체 데이터를 저장
        
        # MyDataset의 입력으로는
        # - 노드 특성: one_hot_nodes
        # - 인접 행렬: 2D 좌표 기반으로 계산된 adj
        # - 엣지 연결 정보: spectral encoding을 적용한 edge_index_2d
        # - 엣지 특성: spectral encoding을 적용한 edge_attr_2d
        # - 타겟: target (다중 라벨 분류)
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
        
        # edge_attr 차원 설정 (spectral encoding된 엣지 특성의 차원)
        self.edge_attr_dim = data["edge_attr_2d"][0].shape[1] if data["edge_attr_2d"] is not None else None
        
        if scaffold_split:
            # pkl 파일에 저장된 "smiles" 정보를 이용해 scaffold split 수행
            smiles_list = data["smiles"]  # 전처리 단계에서 추출한 SMILES 정보
            scaffold_dict = {}
            for i, smi in enumerate(smiles_list):
                murcko = get_murcko_scaffold(smi)
                # Murcko scaffold 계산 실패 시 원본 SMILES를 key로 사용
                key = murcko if murcko is not None else smi
                if key not in scaffold_dict:
                    scaffold_dict[key] = []
                scaffold_dict[key].append(i)

            # scaffold 그룹을 리스트로 변환 (각 그룹: 동일 scaffold를 가진 분자의 인덱스 리스트)
            scaffold_groups = list(scaffold_dict.values())

            # 무작위 셔플 (numpy 대신 파이썬 random.shuffle 사용)
            import random
            random.seed(seed)
            random.shuffle(scaffold_groups)
            
            # 목표 비율 설정: 8:1:1 (train: valid: test)
            total = len(smiles_list)
            n_train = int(total * 0.8)
            n_valid = int(total * 0.1)
            n_test  = total - n_train - n_valid
            
            train_idx, valid_idx, test_idx = [], [], []
            
            # 각 데이터셋별 목표 수량
            targets = {"train": n_train, "valid": n_valid, "test": n_test}
            
            # 각 그룹별 할당: 아직 목표치에 미치지 못한 쪽을 우선으로 할당
            for group in scaffold_groups:
                # 현재 각 split에 남은 할당량 계산
                remaining_train = targets["train"] - len(train_idx)
                remaining_valid = targets["valid"] - len(valid_idx)
                remaining_test  = targets["test"]  - len(test_idx)
                
                remaining = {"train": remaining_train, "valid": remaining_valid, "test": remaining_test}
                # 남은 분자 수가 가장 많은 split에 할당
                target_split = max(remaining, key=remaining.get)
                if target_split == "train":
                    train_idx.extend(group)
                elif target_split == "valid":
                    valid_idx.extend(group)
                else:
                    test_idx.extend(group)
            
            # 최종 결과 저장
            self.indices = {
                "train": train_idx,
                "eval": valid_idx,
                "test": test_idx,
            }
            
            # 디버깅 출력 (옵션)
            print("Total molecules:", total)
            print("Train set size:", len(train_idx))
            print("Valid set size:", len(valid_idx))
            print("Test set size:", len(test_idx))
            print("Combined size:", len(train_idx) + len(valid_idx) + len(test_idx))
