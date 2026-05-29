import argparse
import random
import numpy as np
import torch
import os

# 랜덤 시드 값 설정
seed = 42

# Python 내장 해시 seed 고정
os.environ["PYTHONHASHSEED"] = str(seed)
# cuBLAS 결정론성을 위한 환경 변수 설정
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# 랜덤 시드 고정
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# GPU 연산의 결정론적 결과를 위해 추가 설정
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# PyTorch의 결정론적 알고리즘 사
torch.use_deterministic_algorithms(True)

import wandb
import torch.nn as nn
from data.BBBP.BBBP import BBBPDataset  # BBBP 데이터셋 클래스
from data.BACE.BACE import BACEDataset  # BACE 데이터셋 클래스
from data.SIDER.SIDER import SIDERDataset  # SIDER 데이터셋 클래스
from data.TOX21.TOX21 import Tox21Dataset  # Tox21 데이터셋 클래스
from data.MUV.MUV import MUVDataset  # MUV 데이터셋 클래스
from data.TOXCAST.TOXCAST import ToxCastDataset  # ToxCast 데이터셋 클래스
from data.CLINTOX.CLINTOX import ClintoxDataset  # CLINTOX 데이터셋 클래스

from src.mymodel.layers import MoireLayer, get_moire_focus
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import optim

parser = argparse.ArgumentParser()
parser.add_argument("--LEARNING_RATE", type=float, default=1e-3)
parser.add_argument("--BATCH_SIZE", type=int, default=128)
parser.add_argument("--DROPOUT", type=float, default=0.2)
parser.add_argument("--DEPTH", type=int, default=3)
parser.add_argument("--MLP_DIM", type=int, default=128)
parser.add_argument("--HEADS", type=int, default=8)
parser.add_argument("--T_MAX", type=int, default=150)
parser.add_argument("--WEIGHT_DECAY", type=float, default=1e-4)
parser.add_argument("--SCALE_MIN", type=float, default=0.8)
parser.add_argument("--SCALE_MAX", type=float, default=2.0)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--VERBOSE", type=bool, default=True)
args = parser.parse_args()

# 초기 CONFIG 설정
CONFIG = {
    "MODEL": "ess",
    "DATASET": "BBBP",      # 사용 데이터셋: BBBP
    "DEPTH": args.DEPTH,             
    "MLP_DIM": args.MLP_DIM,         
    "HEADS": args.HEADS,             
    "FOCUS": "gaussian",
    "DROPOUT": args.DROPOUT,         
    "BATCH_SIZE": args.BATCH_SIZE,      
    "LEARNING_RATE": args.LEARNING_RATE,  
    "WEIGHT_DECAY": args.WEIGHT_DECAY,   
    "T_MAX": args.T_MAX,           
    "ETA_MIN": 1e-7,
    "DEVICE": args.device,
    "SCALE_MIN": args.SCALE_MIN,       
    "SCALE_MAX": args.SCALE_MAX,
    "WIDTH_BASE": 1.0,
    "VERBOSE": args.VERBOSE,
}

# 1) DATASET 이름 소문자로
ds_lower = CONFIG["DATASET"].lower()

if ds_lower == "bbbp":
    from utils.exp_bbbp import Aliquot, set_device, set_verbose
elif ds_lower == "bace":
    from utils.exp_bace import Aliquot, set_device, set_verbose
elif ds_lower == "sider":
    from utils.exp_sider import Aliquot, set_device, set_verbose
elif ds_lower == "tox21":
    from utils.exp_tox21 import Aliquot, set_device, set_verbose
elif ds_lower == "muv":
    from utils.exp_muv import Aliquot, set_device, set_verbose
elif ds_lower == "toxcast":
    from utils.exp_toxcast import Aliquot, set_device, set_verbose
elif ds_lower == "clintox":
    from utils.exp_clintox import Aliquot, set_device, set_verbose
else:
    raise ValueError(f"지원하지 않는 DATASET: {CONFIG['DATASET']}")



# wandb sweep로 실행 시 wandb.config에 있는 값으로 덮어쓰기
if wandb.run is not None:
    for key in CONFIG:
        if key in wandb.config:
            CONFIG[key] = wandb.config[key]



set_device(CONFIG["DEVICE"])
dataset = None

match CONFIG["DATASET"]:
    case "BBBP":
        dataset = BBBPDataset(path="/root/2025/sse_moleculenet/data/moleculenet_dataset/data_pkl/bbbp_data.pkl", scaffold_split=True)
        dataset.prediction_size = 1
        criterion = nn.BCEWithLogitsLoss()
    case "BACE":
        dataset = BACEDataset(path="/root/2025/sse_moleculenet/data/BACE/bace_data.pkl", scaffold_split=True)
        dataset.prediction_size = 1  # BACE는 이진 분류 문제
        criterion = nn.BCEWithLogitsLoss()
    case "SIDER":
        dataset = SIDERDataset(path="/root/2025/sse_moleculenet/data/moleculenet_dataset/data_pkl/sider_data_pyg.pkl", scaffold_split=True)
        dataset.prediction_size = 27  # SIDER는 27개의 태스크
        criterion = nn.BCEWithLogitsLoss()
    case "Tox21":
        dataset = Tox21Dataset(path="/root/2025/sse_moleculenet/data/moleculenet_dataset/data_pkl/tox21_data.pkl", scaffold_split=True)
        dataset.prediction_size = 12
        criterion = nn.BCEWithLogitsLoss()
    case "MUV":
        dataset = MUVDataset(path="/root/2025/sse_moleculenet/data/moleculenet_dataset/data_pkl/muv_data_pyg.pkl", scaffold_split=True)
        dataset.prediction_size = 17
        criterion = nn.BCEWithLogitsLoss()
    case "ToxCast":
        dataset = ToxCastDataset(path="/root/2025/sse_moleculenet/data/moleculenet_dataset/data_pkl/toxcast_data_pyg.pkl", scaffold_split=True)
        # 다중 라벨 문제이므로, 첫 샘플의 target 차원에 따라 prediction_size 설정
        dataset.prediction_size = 617
        criterion = nn.BCEWithLogitsLoss()
    case "Clintox": 
        dataset = ClintoxDataset(path="/root/2025/sse_moleculenet/data/moleculenet_dataset/data_pkl/clintox_data.pkl", scaffold_split=False)
        dataset.prediction_size = 2 
        criterion = nn.BCEWithLogitsLoss()
dataset.float()
dataset.batch_size = CONFIG["BATCH_SIZE"]

class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        dims = config["MLP_DIM"]
        self.input = nn.Sequential(
            nn.Linear(dataset.node_feat_size, dims),
            nn.Linear(dims, dims),
        )
        self.layers = nn.ModuleList(
            [
                MoireLayer(
                    input_dim=dims,
                    output_dim=dims,
                    num_heads=config["HEADS"],
                    shift_min=config["SCALE_MIN"],
                    shift_max=config["SCALE_MAX"],
                    dropout=config["DROPOUT"],
                    focus=get_moire_focus(config["FOCUS"]),
                    edge_attr_dim=dataset.edge_attr_dim,
                )
                for _ in range(config["DEPTH"])
            ]
        )
        self.output = nn.Sequential(
            nn.Linear(dims, dims),
            nn.Linear(dims, dataset.prediction_size),
        )

    def forward(self, x, adj, edge_index, edge_attr, mask):
        x = self.input(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, adj, edge_index, edge_attr, mask)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        x = self.output(x)
        return x

model = MyModel(CONFIG)
if CONFIG["DEVICE"] == "cuda":
    model = model.to('cuda')
optimizer = optim.AdamW(
    model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"]
)
scheduler = CosineAnnealingLR(
    optimizer, T_max=CONFIG["T_MAX"], eta_min=CONFIG["ETA_MIN"]
)

aliquot = Aliquot(
    model=model,
    dataset=dataset,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
)(wandb_project=CONFIG["DATASET"], wandb_config=CONFIG, num_epochs=10000, patience=20)

# Training loop는 Aliquot 내부에서 실행
