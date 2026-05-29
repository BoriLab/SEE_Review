# %%
import torch
import wandb
import os
import torch.nn as nn
import numpy as np

VERBOSE = True
MODEL_SAVE_DIR = "best_models_qm9_2"
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
            if "Loss" in k:
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


def compute_mae_metrics(predictions, targets):
    """
    각 타겟(homo, lumo, gap)별 MAE와 전체 평균 MAE를 계산합니다.
    
    Args:
        predictions (torch.Tensor): (batch_size, 3) 형태의 예측값.
        targets (torch.Tensor): (batch_size, 3) 형태의 실제 값.
    
    Returns:
        mae_per_target (torch.Tensor): 각 타겟별 MAE, shape (3,)
        avg_mae (torch.Tensor): 전체 평균 MAE (scalar)
    """
    abs_errors = torch.abs(predictions - targets)
    mae_per_target = abs_errors.mean(dim=0)
    avg_mae = mae_per_target.mean()
    return mae_per_target, avg_mae


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
        y_hat = self.model(*bat[:-1])
        loss = self.criterion(y_hat, y)
        return loss

    def eval(self):
        self.model.eval()
        total_eval_loss = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for tup in self.eval_dataloader:
                bat = [tensor.to(self.device) for tensor in tup]
                y = bat[-1]
                y_hat = self.model(*bat[:-1])
                loss = self.criterion(y_hat, y)
                batch_size = y.size(0)
                total_eval_loss += loss.item() * batch_size
                total_samples += batch_size
                all_preds.append(y_hat)
                all_targets.append(y)
        avg_eval_loss = total_eval_loss / total_samples
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mae_per_target, avg_mae = compute_mae_metrics(all_preds, all_targets)
        _wandb_log({
            "Loss/eval": avg_eval_loss,
            "MAE_per_target": mae_per_target.tolist(),
            "Avg_MAE": avg_mae.item()
        })
        return avg_eval_loss

    def test(self):
        self.model = load_model(self.model, wandb.run.name + ".pth", device=self.device)
        self.model.eval()
        total_test_loss = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for tup in self.test_dataloader:
                bat = [tensor.to(self.device) for tensor in tup]
                y = bat[-1]
                y_hat = self.model(*bat[:-1])
                loss = self.criterion(y_hat, y)
                batch_size = y.size(0)
                total_test_loss += loss.item() * batch_size
                total_samples += batch_size
                all_preds.append(y_hat)
                all_targets.append(y)
        avg_test_loss = total_test_loss / total_samples
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mae_per_target, avg_mae = compute_mae_metrics(all_preds, all_targets)
        _wandb_log({
            "Loss/test": avg_test_loss,
            "Test_MAE_per_target": mae_per_target.tolist(),
            "Test_Avg_MAE": avg_mae.item()
        })
        return avg_test_loss

    def train(self, num_epochs=10000, patience=20):
        best_eval_loss = float("inf")
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
            # 평균 train loss (학습 중 기록)
            avg_train_loss = total_train_loss / total_samples

            # 에포크 종료 후, train 데이터셋 전체에 대한 MAE 평가
            self.model.eval()
            train_total_loss = 0
            train_total_samples = 0
            train_preds = []
            train_targets = []
            with torch.no_grad():
                for tup in self.train_dataloader:
                    bat = [tensor.to(self.device) for tensor in tup]
                    y = bat[-1]
                    y_hat = self.model(*bat[:-1])
                    loss = self.criterion(y_hat, y)
                    bs = y.size(0)
                    train_total_loss += loss.item() * bs
                    train_total_samples += bs
                    train_preds.append(y_hat)
                    train_targets.append(y)
            avg_train_loss_eval = train_total_loss / train_total_samples
            train_preds = torch.cat(train_preds, dim=0)
            train_targets = torch.cat(train_targets, dim=0)
            train_mae_per_target, train_avg_mae = compute_mae_metrics(train_preds, train_targets)
            _wandb_log({
                "Epoch": epoch,
                "Loss/train": avg_train_loss_eval,
                "Train_MAE_per_target": train_mae_per_target.tolist(),
                "Train_Avg_MAE": train_avg_mae.item()
            })

            eval_loss = self.eval()
            _wandb_log({"Loss/eval": eval_loss})
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_model(self.model)
                epochs_without_improvement = 0
                _wandb_log({"Loss/best_eval": best_eval_loss})
                wandb.run.summary["best_eval_loss"] = best_eval_loss
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement > patience:
                break
        return best_eval_loss

    def __call__(self, wandb_project="", wandb_config={}, num_epochs=10000, patience=30):
        self.one_batch_check()
        if wandb_project:
            wandb.init(
                project=wandb_project,
                config=wandb_config,
                settings=wandb.Settings(code_dir="."),
            )
        self.train(num_epochs, patience)
        test_loss = self.test()
        _wandb_log({"Loss/test": test_loss})
        if wandb_project:
            wandb.finish()
        return test_loss

    def one_batch_check(self):
        self.model.train()
        for tup in self.train_dataloader:
            self.optimizer.zero_grad()
            tup = [tensor.to(self.device) for tensor in tup]
            y = tup[-1]
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
        shift_means = []
        width_means = []
        shift_values = []  # 모든 레이어의 shift 값을 저장할 리스트입니다.
        width_values = []  # 모든 레이어의 width 값을 저장할 리스트입니다.
        for i, layer in enumerate(model.layers):
            attention_layer = layer.attention
            shift_values.append(attention_layer.shifts.detach().cpu().numpy())
            width_values.append(attention_layer.widths.detach().cpu().numpy())
            shift_mean = attention_layer.shifts.detach().cpu().mean().item()
            width_mean = attention_layer.widths.detach().cpu().mean().item()
            shift_means.append(shift_mean)
            width_means.append(width_mean)
            monitoring_dict[f"shift_mean_layer_{i}"] = shift_mean
            monitoring_dict[f"width_mean_layer_{i}"] = width_mean
        monitoring_dict["shift_values_histogram"] = wandb.Histogram(np.array(shift_values))
        monitoring_dict["width_values_histogram"] = wandb.Histogram(np.array(width_values))
        _wandb_log(monitoring_dict)
        return monitoring_dict
