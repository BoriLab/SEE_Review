from utils.dataset_clintox import MyDataset
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

class ClintoxDataset(MyDataset):
    def __init__(
        self,
        path="",
        evaluation_size=0.1,
        test_size=0.1,
        batch_size=128,
        seed=42,
        scaffold_split=True  # scaffold 분할 사용 여부
    ):
        # 피클 파일로부터 전처리된 Clintox 데이터를 로드
        data = pickle.load(open(path, "rb"))
        # 만약 data가 리스트라면, key별로 모아서 딕셔너리 형태로 변환
        if isinstance(data, list):
            keys = data[0].keys()
            data_dict = {key: [sample[key] for sample in data] for key in keys}
            data = data_dict
        self.data = data  # scaffold 분할에 사용하기 위해 저장
        
        # MyDataset의 입력 인자에 맞게 데이터를 전달합니다.
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
        
        # edge_attr_dim 설정 (spectral encoding된 엣지 특성의 차원)
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
            frac_train = 1 - (evaluation_size + test_size)
            frac_valid = evaluation_size
            total = len(smiles_list)
            # GEM 방식 cutoff 계산: 학습(train) 및 학습+평가(valid)까지의 분자 수 기준
            train_cutoff = int(frac_train * total)
            valid_cutoff = int((frac_train + frac_valid) * total)
            
            train_idx, valid_idx, test_idx = [], [], []
            
            # 각 scaffold 그룹별로 그룹 전체를 train, valid, test 중 하나에 할당
            for group in scaffold_groups:
                if len(train_idx) + len(group) > train_cutoff:
                    if len(train_idx) + len(valid_idx) + len(group) > valid_cutoff:
                        test_idx.extend(group)
                    else:
                        valid_idx.extend(group)
                else:
                    train_idx.extend(group)
            
            self.indices = {
                "train": train_idx,
                "eval": valid_idx,
                "test": test_idx,
            }