import torch
import wandb
import os
import torch.nn as nn
import math
import numpy as np
from sklearn.metrics import roc_auc_score

VERBOSE = True
MODEL_SAVE_DIR = "best_models_toxcast"
DEVICE = ""

def set_verbose(verbose):
    global VERBOSE
    VERBOSE = verbose

def set_model_save_dir(model_save_dir):
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    global MODEL_SAVE_DIR
    MODEL_SAVE_DIR = model_save_dir

def save_model(model):
    path = wandb.run.name + ".pth"
    if MODEL_SAVE_DIR is not None:
        path = os.path.join(MODEL_SAVE_DIR, path)
        if not os.path.exists(MODEL_SAVE_DIR):
            os.makedirs(MODEL_SAVE_DIR)
    torch.save(model.state_dict(), path)

def _wandb_log(dic):
    if len(dic) == 0:
        return
    if wandb.run is not None:
        try:
            wandb.log(dic)
        except Exception as e:
            print(e)
    if VERBOSE:
        for k in dic:
            if "Loss" in k or "ROC_AUC" in k:
                print(dic)
                return

def set_device(device):
    global DEVICE
    DEVICE = device

def get_device():
    if DEVICE:
        return DEVICE
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if VERBOSE:
        print(f"Using device: {device}")
    return device

def load_model(model, path, device=get_device()):
    if MODEL_SAVE_DIR is not None:
        path = os.path.join(MODEL_SAVE_DIR, path)
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model

def compute_roc_auc_metrics(predictions, targets):
    """
    예측값(logits)에 대해 sigmoid를 적용한 후, 
    NaN이 아닌 타겟 값만을 대상으로 ROC-AUC를 계산합니다.
    
    Args:
        predictions (torch.Tensor): (batch_size, num_tasks) 형태의 예측값.
        targets (torch.Tensor): (batch_size, num_tasks) 형태의 실제 값.
        
    Returns:
        auc (float): ROC-AUC 스코어
    """
    valid_mask = ~torch.isnan(targets)
    if valid_mask.sum() == 0:
        return 0.5
    preds_prob = torch.sigmoid(predictions)[valid_mask].detach().cpu().numpy().flatten()
    targets_np = targets[valid_mask].detach().cpu().numpy().flatten()
    try:
        auc = roc_auc_score(targets_np, preds_prob)
    except Exception as e:
        auc = 0.5  # 예외 발생 시 기본값
    return auc

class Aliquot:
    def __init__(self, model, dataset, optimizer, criterion, scheduler=None):
        self.device = get_device()
        self.model = model.to(self.device).float()
        self.dataset = dataset
        self.train_dataloader = dataset.train()
        self.eval_dataloader = dataset.eval()
        self.test_dataloader = dataset.test()
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _loop(self, bat):
        # 배치 내 모든 텐서를 device로 이동
        bat = [tensor.to(self.device) for tensor in bat]
        y = bat[-1]
        if y.dim() == 3 and y.size(1) == 1:
            y = y.squeeze(1)
        y_hat = self.model(*bat[:-1])
        # NaN이 아닌 target 위치만을 사용하여 손실 계산
        valid_mask = ~torch.isnan(y)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        loss = self.criterion(y_hat[valid_mask], y[valid_mask])
        return loss

    def eval(self):
        self.model.eval()
        total_eval_loss = 0.0
        total_samples = 0
        eval_preds = []
        eval_targets = []
        with torch.no_grad():
            for tup in self.eval_dataloader:
                bat = [tensor.to(self.device) for tensor in tup]
                y = bat[-1]
                if y.dim() == 3 and y.size(1) == 1:
                    y = y.squeeze(1)
                y_hat = self.model(*bat[:-1])
                valid_mask = ~torch.isnan(y)
                if valid_mask.sum() > 0:
                    loss = self.criterion(y_hat[valid_mask], y[valid_mask])
                else:
                    loss = torch.tensor(0.0, device=self.device)
                batch_size = y.size(0)
                total_eval_loss += loss.item() * batch_size
                total_samples += batch_size
                # 유효한 데이터만 저장
                if valid_mask.sum() > 0:
                    eval_preds.append(y_hat[valid_mask])
                    eval_targets.append(y[valid_mask])
        avg_eval_loss = total_eval_loss / total_samples
        if eval_preds:
            eval_preds = torch.cat(eval_preds, dim=0)
            eval_targets = torch.cat(eval_targets, dim=0)
            eval_auc = compute_roc_auc_metrics(eval_preds, eval_targets)
        else:
            eval_auc = 0.5
        _wandb_log({
            "Loss/eval": avg_eval_loss,
            "ROC_AUC/eval": eval_auc,
        })
        return eval_auc, avg_eval_loss

    def test(self):
        self.model = load_model(self.model, wandb.run.name + ".pth", device=self.device)
        self.model.eval()
        total_test_loss = 0.0
        total_samples = 0
        test_preds = []
        test_targets = []
        with torch.no_grad():
            for tup in self.test_dataloader:
                bat = [tensor.to(self.device) for tensor in tup]
                y = bat[-1]
                if y.dim() == 3 and y.size(1) == 1:
                    y = y.squeeze(1)
                y_hat = self.model(*bat[:-1])
                valid_mask = ~torch.isnan(y)
                if valid_mask.sum() > 0:
                    loss = self.criterion(y_hat[valid_mask], y[valid_mask])
                else:
                    loss = torch.tensor(0.0, device=self.device)
                batch_size = y.size(0)
                total_test_loss += loss.item() * batch_size
                total_samples += batch_size
                if valid_mask.sum() > 0:
                    test_preds.append(y_hat[valid_mask])
                    test_targets.append(y[valid_mask])
        avg_test_loss = total_test_loss / total_samples
        if test_preds:
            test_preds = torch.cat(test_preds, dim=0)
            test_targets = torch.cat(test_targets, dim=0)
            test_auc = compute_roc_auc_metrics(test_preds, test_targets)
        else:
            test_auc = 0.5
        _wandb_log({
            "Loss/test": avg_test_loss,
            "Test_ROC_AUC": test_auc,
        })
        return test_auc

    def train(self, num_epochs=10000, patience=20):
        best_auc = 0.0  # ROC-AUC는 높을수록 좋음
        epochs_without_improvement = 0
        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0.0
            total_samples = 0
            for step, tup in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                loss = self._loop(tup)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                batch_size = tup[0].size(0)
                total_train_loss += loss.item() * batch_size
                total_samples += batch_size
                if step % 10 == 0:
                    _wandb_log({"Loss/train_step": loss.item()})
                    _wandb_log(self._get_monitoring_parameters())
            avg_train_loss = total_train_loss / total_samples

            # Train 데이터에 대해 ROC-AUC 계산
            self.model.eval()
            train_preds = []
            train_targets = []
            with torch.no_grad():
                for tup in self.train_dataloader:
                    bat = [tensor.to(self.device) for tensor in tup]
                    y = bat[-1]
                    if y.dim() == 3 and y.size(1) == 1:
                        y = y.squeeze(1)
                    y_hat = self.model(*bat[:-1])
                    train_preds.append(y_hat)
                    train_targets.append(y)
            train_preds = torch.cat(train_preds, dim=0)
            train_targets = torch.cat(train_targets, dim=0)
            train_auc = compute_roc_auc_metrics(train_preds, train_targets)
            _wandb_log({
                "Epoch": epoch,
                "Loss/train": avg_train_loss,
                "Train_ROC_AUC": train_auc
            })

            # Evaluation 단계
            eval_auc, avg_eval_loss = self.eval()

            if eval_auc > best_auc:
                best_auc = eval_auc
                save_model(self.model)
                epochs_without_improvement = 0
                _wandb_log({"Best_ROC_AUC": best_auc})
                wandb.run.summary["best_eval_auc"] = best_auc
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement > patience:
                break
        return best_auc

    def __call__(self, wandb_project="", wandb_config={}, num_epochs=10000, patience=30):
        self.one_batch_check()
        if wandb_project:
            wandb.init(
                project=wandb_project,
                config=wandb_config,
                settings=wandb.Settings(code_dir="."),
            )
        best_auc = self.train(num_epochs, patience)
        test_auc = self.test()
        _wandb_log({"Test_ROC_AUC": test_auc})
        if wandb_project:
            wandb.finish()
        return test_auc

    def one_batch_check(self):
        self.model.train()
        for tup in self.train_dataloader:
            self.optimizer.zero_grad()
            tup = [tensor.to(self.device) for tensor in tup]
            y = tup[-1]
            if y.dim() == 3 and y.size(1) == 1:
                y = y.squeeze(1)
            y_hat = self.model(*tup[:-1])
            valid_mask = ~torch.isnan(y)
            if valid_mask.sum() == 0:
                loss = torch.tensor(0.0, device=self.device)
            else:
                loss = self.criterion(y_hat[valid_mask], y[valid_mask])
            loss.backward()
            self.optimizer.step()
            # Sanity Check
            assert y_hat is not None, "Model output is None"
            assert not torch.isnan(y_hat).any(), "Model output has NaN"
            assert not torch.isinf(y_hat).any(), "Model output has inf"
            assert not torch.isnan(loss).any(), "Loss has NaN"
            assert not torch.isinf(loss).any(), "Loss has inf"
            break

    def _get_monitoring_parameters(self):
        monitoring_dict = {}
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(model, 'layers'):
            shift_values = []
            width_values = []
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'attention'):
                    attention_layer = layer.attention
                    shift_values.append(attention_layer.shifts.detach().cpu().numpy())
                    width_values.append(attention_layer.widths.detach().cpu().numpy())
                    shift_mean = attention_layer.shifts.detach().cpu().mean().item()
                    width_mean = attention_layer.widths.detach().cpu().mean().item()
                    monitoring_dict[f"shift_mean_layer_{i}"] = shift_mean
                    monitoring_dict[f"width_mean_layer_{i}"] = width_mean
            monitoring_dict["shift_values_histogram"] = wandb.Histogram(np.array(shift_values))
            monitoring_dict["width_values_histogram"] = wandb.Histogram(np.array(width_values))
        _wandb_log(monitoring_dict)
        return monitoring_dict


#-------청크

# import os
# import random
# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import wandb
# from sklearn.metrics import roc_auc_score
# from torch.optim.lr_scheduler import CosineAnnealingLR

# VERBOSE = True
# MODEL_SAVE_DIR = "best_models_toxcastpyg"
# DEVICE = ""

# def set_verbose(verbose):
#     global VERBOSE
#     VERBOSE = verbose

# def set_model_save_dir(model_save_dir):
#     if not os.path.exists(model_save_dir):
#         os.makedirs(model_save_dir)
#     global MODEL_SAVE_DIR
#     MODEL_SAVE_DIR = model_save_dir

# def save_model(model):
#     path = wandb.run.name + ".pth"
#     if MODEL_SAVE_DIR is not None:
#         path = os.path.join(MODEL_SAVE_DIR, path)
#         if not os.path.exists(MODEL_SAVE_DIR):
#             os.makedirs(MODEL_SAVE_DIR)
#     torch.save(model.state_dict(), path)

# def _wandb_log(dic):
#     if not dic:
#         return
#     if wandb.run is not None:
#         try:
#             wandb.log(dic)
#         except Exception as e:
#             print(e)
#     if VERBOSE:
#         for k in dic:
#             if "Loss" in k or "ROC_AUC" in k:
#                 print(f"{k}: {dic[k]}")
#                 return

# def set_device(device):
#     global DEVICE
#     DEVICE = device

# def get_device():
#     if DEVICE:
#         return DEVICE
#     device = torch.device(
#         "mps" if torch.backends.mps.is_available()
#         else "cuda" if torch.cuda.is_available()
#         else "cpu"
#     )
#     if VERBOSE:
#         print(f"Using device: {device}")
#     return device

# def load_model(model, path, device=None):
#     if device is None:
#         device = get_device()
#     if MODEL_SAVE_DIR is not None:
#         path = os.path.join(MODEL_SAVE_DIR, path)
#     model.load_state_dict(torch.load(path, map_location=device))
#     model.to(device)
#     return model

# def compute_roc_auc_metrics(preds, targets):
#     valid = ~torch.isnan(targets)
#     if valid.sum() == 0:
#         return 0.5
#     p = torch.sigmoid(preds)[valid].cpu().numpy().flatten()
#     t = targets[valid].cpu().numpy().flatten()
#     try:
#         return roc_auc_score(t, p)
#     except:
#         return 0.5

# class Aliquot:
#     def __init__(self, model, dataset, optimizer, criterion, scheduler=None, n_splits=3):
#         self.device    = get_device()
#         self.model     = model.to(self.device).float()
#         self.dataset   = dataset
#         self.train_dl  = dataset.train()
#         self.eval_dl   = dataset.eval()
#         self.test_dl   = dataset.test()
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.scheduler = scheduler
#         self.n_splits  = n_splits

#     def compute_chunked_loss(self, logits, targets):
#         total_tasks = logits.size(1)
#         chunk_size  = math.ceil(total_tasks / self.n_splits)
#         loss = 0.0
#         for i in range(self.n_splits):
#             start = i * chunk_size
#             end   = min(start + chunk_size, total_tasks)
#             if start >= end:
#                 break
#             logit_chunk  = logits[:, start:end]
#             target_chunk = targets[:, start:end]
#             valid = ~torch.isnan(target_chunk)
#             if valid.sum() > 0:
#                 loss_i = self.criterion(
#                     logit_chunk[valid], target_chunk[valid]
#                 )
#                 loss  += loss_i
#         return loss  # or loss / self.n_splits for averaged

#     def _loop(self, batch):
#         batch = [t.to(self.device) for t in batch]
#         y = batch[-1]
#         if y.dim() == 3 and y.size(1) == 1:
#             y = y.squeeze(1)
#         logits = self.model(*batch[:-1])
#         return self.compute_chunked_loss(logits, y)

#     def train(self, num_epochs=10000, patience=20):
#         best_auc = 0.0
#         no_improve = 0
#         for epoch in range(num_epochs):
#             self.model.train()
#             total_loss, total_samples = 0.0, 0
#             for batch in self.train_dl:
#                 self.optimizer.zero_grad()
#                 loss = self._loop(batch)
#                 loss.backward()
#                 self.optimizer.step()
#                 if self.scheduler:
#                     self.scheduler.step()
#                 bs = batch[0].size(0)
#                 total_loss   += loss.item() * bs
#                 total_samples+= bs
#             avg_train_loss = total_loss / total_samples

#             # compute train AUC
#             train_preds, train_targets = [], []
#             self.model.eval()
#             with torch.no_grad():
#                 for batch in self.train_dl:
#                     batch = [t.to(self.device) for t in batch]
#                     y = batch[-1]
#                     if y.dim() == 3 and y.size(1) == 1:
#                         y = y.squeeze(1)
#                     logits = self.model(*batch[:-1])
#                     mask = ~torch.isnan(y)
#                     train_preds.append(logits[mask])
#                     train_targets.append(y[mask])
#             train_auc = compute_roc_auc_metrics(
#                 torch.cat(train_preds), torch.cat(train_targets)
#             )
#             _wandb_log({
#                 "Epoch": epoch,
#                 "Loss/train": avg_train_loss,
#                 "Train_ROC_AUC": train_auc
#             })

#             eval_auc, eval_loss = self.eval()
#             if eval_auc > best_auc:
#                 best_auc = eval_auc
#                 save_model(self.model)
#                 no_improve = 0
#                 _wandb_log({"Best_ROC_AUC": best_auc})
#                 wandb.run.summary["best_eval_auc"] = best_auc
#             else:
#                 no_improve += 1
#                 if no_improve > patience:
#                     break

#         return best_auc

#     def eval(self):
#         self.model.eval()
#         total_loss, total_samples = 0.0, 0
#         preds, targets = [], []
#         with torch.no_grad():
#             for batch in self.eval_dl:
#                 batch = [t.to(self.device) for t in batch]
#                 y = batch[-1]
#                 if y.dim() == 3 and y.size(1) == 1:
#                     y = y.squeeze(1)
#                 logits = self.model(*batch[:-1])
#                 loss = self.compute_chunked_loss(logits, y)
#                 bs = y.size(0)
#                 total_loss   += loss.item() * bs
#                 total_samples+= bs
#                 mask = ~torch.isnan(y)
#                 preds.append(logits[mask])
#                 targets.append(y[mask])
#         avg_loss = total_loss / total_samples
#         if preds:
#             eval_auc = compute_roc_auc_metrics(
#                 torch.cat(preds), torch.cat(targets)
#             )
#         else:
#             eval_auc = 0.5
#         _wandb_log({"Loss/eval": avg_loss, "ROC_AUC/eval": eval_auc})
#         return eval_auc, avg_loss

#     def test(self):
#         load_model(self.model, wandb.run.name + ".pth", device=self.device)
#         self.model.eval()
#         total_loss, total_samples = 0.0, 0
#         preds, targets = [], []
#         with torch.no_grad():
#             for batch in self.test_dl:
#                 batch = [t.to(self.device) for t in batch]
#                 y = batch[-1]
#                 if y.dim() == 3 and y.size(1) == 1:
#                     y = y.squeeze(1)
#                 logits = self.model(*batch[:-1])
#                 loss = self.compute_chunked_loss(logits, y)
#                 bs = y.size(0)
#                 total_loss   += loss.item() * bs
#                 total_samples+= bs
#                 mask = ~torch.isnan(y)
#                 preds.append(logits[mask])
#                 targets.append(y[mask])
#         avg_loss = total_loss / total_samples
#         if preds:
#             test_auc = compute_roc_auc_metrics(
#                 torch.cat(preds), torch.cat(targets)
#             )
#         else:
#             test_auc = 0.5
#         _wandb_log({"Loss/test": avg_loss, "ROC_AUC/test": test_auc})
#         return test_auc

#     def __call__(self, wandb_project="", wandb_config={}, num_epochs=10000, patience=20):
#         # sanity check on one batch
#         self.model.train()
#         for batch in self.train_dl:
#             self.optimizer.zero_grad()
#             batch = [t.to(self.device) for t in batch]
#             y = batch[-1]
#             if y.dim()==3 and y.size(1)==1:
#                 y = y.squeeze(1)
#             logits = self.model(*batch[:-1])
#             loss = self.compute_chunked_loss(logits, y)
#             loss.backward()
#             self.optimizer.step()
#             break

#         if wandb_project:
#             wandb.init(
#                 project=wandb_project,
#                 config=wandb_config,
#                 settings=wandb.Settings(code_dir="."),
#             )
#         best_auc = self.train(num_epochs, patience)
#         test_auc = self.test()
#         _wandb_log({"Final_Test_ROC_AUC": test_auc})
#         if wandb_project:
#             wandb.finish()
#         return test_auc
