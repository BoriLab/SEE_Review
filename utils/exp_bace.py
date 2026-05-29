import torch
import wandb
import os
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score  # ROC-AUC 계산을 위한 import

VERBOSE = True
MODEL_SAVE_DIR = "best_models"
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
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
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
    예측값(logits)에 대해 sigmoid를 적용한 후, ROC-AUC를 계산합니다.
    (BACE 데이터셋: 이진 분류)
    
    Args:
        predictions (torch.Tensor): (batch_size, 1) 형태의 예측값.
        targets (torch.Tensor): (batch_size, 1) 형태의 실제 값.
        
    Returns:
        auc (float): ROC-AUC 스코어
    """
    preds_prob = torch.sigmoid(predictions).detach().cpu().numpy().flatten()
    targets_np = targets.detach().cpu().numpy().flatten()
    try:
        auc = roc_auc_score(targets_np, preds_prob)
    except Exception as e:
        auc = 0.5  # 예외 상황 처리
    return auc


class Aliquot:
    def __init__(
        self,
        model,
        dataset,
        optimizer,
        criterion,
        scheduler=None,
    ):
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
        bat = [tensor.to(self.device) for tensor in bat]
        y = bat[-1]
        # BACE 데이터셋의 타겟 shape가 [batch, 1, 1]이면 [batch, 1]로 squeeze
        if y.dim() == 3:
            y = y.squeeze(-1)
        y_hat = self.model(*bat[:-1])
        loss = self.criterion(y_hat, y)
        return loss

    def eval(self):
        self.model.eval()
        total_eval_loss = 0
        total_samples = 0
        eval_preds = []
        eval_targets = []
        with torch.no_grad():
            for tup in self.eval_dataloader:
                bat = [tensor.to(self.device) for tensor in tup]
                y = bat[-1]
                if y.dim() == 3:
                    y = y.squeeze(-1)
                y_hat = self.model(*bat[:-1])
                loss = self.criterion(y_hat, y)
                batch_size = y.size(0)
                total_eval_loss += loss.item() * batch_size
                total_samples += batch_size
                eval_preds.append(y_hat)
                eval_targets.append(y)
        avg_eval_loss = total_eval_loss / total_samples
        eval_preds = torch.cat(eval_preds, dim=0)
        eval_targets = torch.cat(eval_targets, dim=0)
        eval_auc = compute_roc_auc_metrics(eval_preds, eval_targets)
        _wandb_log({
            "Loss/eval": avg_eval_loss,
            "ROC_AUC/eval": eval_auc,
        })
        return eval_auc, avg_eval_loss

    def test(self):
        self.model = load_model(self.model, wandb.run.name + ".pth", device=self.device)
        self.model.eval()
        total_test_loss = 0
        total_samples = 0
        test_preds = []
        test_targets = []
        with torch.no_grad():
            for tup in self.test_dataloader:
                bat = [tensor.to(self.device) for tensor in tup]
                y = bat[-1]
                if y.dim() == 3:
                    y = y.squeeze(-1)
                y_hat = self.model(*bat[:-1])
                loss = self.criterion(y_hat, y)
                batch_size = y.size(0)
                total_test_loss += loss.item() * batch_size
                total_samples += batch_size
                test_preds.append(y_hat)
                test_targets.append(y)
        avg_test_loss = total_test_loss / total_samples
        test_preds = torch.cat(test_preds, dim=0)
        test_targets = torch.cat(test_targets, dim=0)
        test_auc = compute_roc_auc_metrics(test_preds, test_targets)
        _wandb_log({
            "Loss/test": avg_test_loss,
            "Test_ROC_AUC": test_auc,
        })
        return test_auc

    def train(self, num_epochs=10000, patience=20):
        best_auc = 0.0  # ROC-AUC는 높을수록 좋으므로 초기값 0.0
        epochs_without_improvement = 0
        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0
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

            self.model.eval()
            train_preds = []
            train_targets = []
            with torch.no_grad():
                for tup in self.train_dataloader:
                    bat = [tensor.to(self.device) for tensor in tup]
                    y = bat[-1]
                    if y.dim() == 3:
                        y = y.squeeze(-1)
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

            eval_auc, avg_eval_loss = self.eval()

            if eval_auc > best_auc:
                best_auc = eval_auc
                save_model(self.model)
                epochs_without_improvement = 0
                _wandb_log({"Best_eval_ROC_AUC": best_auc})
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
            # BACE 데이터셋의 경우, 타겟의 shape가 [batch, 1, 1]이면 [batch, 1]로 변경
            if y.dim() == 3:
                y = y.squeeze(-1)
            y_hat = self.model(*tup[:-1])
            loss = self.criterion(y_hat, y)
            loss.backward()
            self.optimizer.step()
            # Sanity check
            assert y_hat is not None, "Model output is None"
            assert not torch.isnan(y_hat).any(), "Model output has NaN"
            assert not torch.isinf(y_hat).any(), "Model output has inf"
            assert not torch.isnan(loss).any(), "Loss has NaN"
            assert not torch.isinf(loss).any(), "Loss has inf"
            break

    def _get_monitoring_parameters(self):
        monitoring_dict = {}
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        # 만약 모델에 layers 속성이 없으면 빈 dict 반환
        if not hasattr(model, "layers"):
            return monitoring_dict
        shift_values = []
        width_values = []
        for i, layer in enumerate(model.layers):
            if not hasattr(layer, "attention"):
                continue
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
