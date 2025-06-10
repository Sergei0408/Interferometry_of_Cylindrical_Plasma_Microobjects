import os

import lightning as L
import pandas as pd
import torch
from torch import optim
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score
from transformers import get_cosine_schedule_with_warmup

from src.losses import TorchLoss
from src.models import TorchModel


class TrainPipeline(L.LightningModule):
    def __init__(self, config, train_loader, val_loader) -> None:
        super().__init__()
        self.config = config

        self.model = TorchModel(**config['model'])
        if config['weights'] is not None:
            state_dict = torch.load(config['weights'], map_location='cpu')['state_dict']
            self.load_state_dict(state_dict, strict=True)

        # Freeze all parameters first
        for p in self.model.model.parameters():
            p.requires_grad = False

        self.tb = config['trainable_blocks']
        if self.tb == 'head':
            # Unfreeze only the classifier head
            try:
                for name in ('fc', 'head', 'classifier'):
                    if hasattr(self.model.model, name):
                        head = getattr(self.model.model, name)
                        break
                else:
                    head = self.model.model.get_classifier()
            except AttributeError:
                raise RuntimeError("Не удалось найти модуль-голову. Уточните имя подмодуля в модели.")
            for p in head.parameters():
                p.requires_grad = True

        elif isinstance(self.tb, int) and self.tb > 0:
            # Unfreeze only the classifier head
            try:
                for name in ('fc', 'head', 'classifier'):
                    if hasattr(self.model.model, name):
                        head = getattr(self.model.model, name)
                        break
                else:
                    head = self.model.model.get_classifier()
            except AttributeError:
                raise RuntimeError("Не удалось найти модуль-голову. Уточните имя подмодуля в модели.")
            for p in head.parameters():
                p.requires_grad = True

            # Unfreeze N last blocks
            if hasattr(self.model.model, 'stages') and len(self.model.model.stages) > 0:
                last_stage = self.model.model.stages[-1]
                if hasattr(last_stage, 'blocks'):
                    all_blocks = list(last_stage.blocks)
                else:
                    # fallback: если нет атрибута blocks, пробуем отдать сам stage как iterable
                    all_blocks = list(last_stage)

                for block in all_blocks[-self.tb:]:
                    for p in block.parameters():
                        p.requires_grad = True


            else:
                children = [m for _, m in self.model.model.named_children()]
                for module in children[-self.tb:]:
                    for p in module.parameters():
                        p.requires_grad = True


        # Все остальные случаи — разморозить всё
        else:
            for p in self.model.model.parameters():
                p.requires_grad = True


        self.criterion = TorchLoss()
        metrics = MetricCollection([            
            MeanSquaredError(),
            MeanAbsoluteError(),
            R2Score()
            ])
        
        self.train_metrics = metrics.clone(postfix="/train")
        self.valid_metrics = metrics.clone(postfix="/val")

        self.train_loader = train_loader
        self.val_loader = val_loader
        # In case of DDP
        # self.num_training_steps = math.ceil(len(self.train_loader) / len(config['trainer']['devices']))
        self.num_training_steps = len(self.train_loader)

        self.save_hyperparameters(config)

    def configure_optimizers(self):
        if self.config['optimizer'] == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                **self.config['optimizer_params']
            )
        elif self.config['optimizer'] == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                momentum=0.9, nesterov=True,
                **self.config['optimizer_params']
            )
        else:
            raise ValueError(f"Unknown optimizer name: {self.config['optimizer']}")

        scheduler_params = self.config['scheduler_params']
        if self.hparams.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=scheduler_params['patience'],
                min_lr=scheduler_params['min_lr'],
                factor=scheduler_params['factor'],
                mode=scheduler_params['mode'],
                verbose=scheduler_params['verbose'],
            )

            lr_scheduler = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': scheduler_params['target_metric']
            }
        elif self.config['scheduler'] == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.num_training_steps * scheduler_params['warmup_epochs'],
                num_training_steps=int(self.num_training_steps * self.config['trainer']['max_epochs'])
            )

            lr_scheduler = {
                'scheduler': scheduler,
                'interval': 'step'
            }
        else:
            raise ValueError(f"Unknown scheduler name: {self.config['scheduler']}")

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)

        self.log("Loss/train", loss, prog_bar=True)
        self.train_metrics.update(out, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        self.log("Loss/val", loss, prog_bar=True)
        self.valid_metrics.update(out, y)

    def on_train_epoch_end(self):
        train_metrics = self.train_metrics.compute()
        self.log_dict(train_metrics)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        valid_metrics = self.valid_metrics.compute()
        self.log_dict(valid_metrics)
        self.valid_metrics.reset()


class TestPipeline(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = TorchModel(**config['model'])
        state = torch.load(config['weights'], map_location='cpu')
        self.load_state_dict(state.get('state_dict', state), strict=True)


        self.preds = []
        self.targets = []
        self.idxs = []


        self.means = torch.tensor(config['target_cols_means'], dtype=torch.float32)
        self.stds  = torch.tensor(config['target_cols_stds'],  dtype=torch.float32)

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch):
        x, y, idx = batch['image'], batch['label'], batch['idx']
        out = self.model(x)

        self.preds.append(out.detach().cpu())
        self.targets.append(y.detach().cpu())
        self.idxs.append(idx.detach().cpu())

    def on_test_epoch_end(self):

        preds   = torch.cat(self.preds,   dim=0)
        targets = torch.cat(self.targets, dim=0)
        idxs    = torch.cat(self.idxs,     dim=0)

        norm_metrics = {}
        for i, col in enumerate(self.config['target_cols']):
            mse = MeanSquaredError()(preds[:, i], targets[:, i])
            mae = MeanAbsoluteError()(preds[:, i], targets[:, i])
            r2  = R2Score()(preds[:, i], targets[:, i])
            norm_metrics[f"{col}/MSE"] = mse.item()
            norm_metrics[f"{col}/MAE"] = mae.item()
            norm_metrics[f"{col}/R2"]  = r2.item()

        self.log_dict(norm_metrics, prog_bar=True)

        save_dir = os.path.join(self.config['save_path'], 'predictions')
        os.makedirs(save_dir, exist_ok=True)
        norm_file = os.path.join(save_dir, 'metrics_norm.txt')
        with open(norm_file, 'w', encoding='utf-8') as f:
            f.write("Нормированные метрики по колонкам:\n")
            for name, value in norm_metrics.items():
                f.write(f"{name}: {value:.6f}\n")

        preds_den = preds * self.stds + self.means
        targ_den  = targets * self.stds + self.means

        denorm_metrics = {}
        for i, col in enumerate(self.config['target_cols']):
            mse = MeanSquaredError()(preds_den[:, i], targ_den[:, i])
            mae = MeanAbsoluteError()(preds_den[:, i], targ_den[:, i])
            r2  = R2Score()(preds_den[:, i], targ_den[:, i])
            denorm_metrics[f"{col}/MSE"] = mse.item()
            denorm_metrics[f"{col}/MAE"] = mae.item()
            denorm_metrics[f"{col}/R2"]  = r2.item()


        self.log_dict(denorm_metrics, prog_bar=True)
        denorm_file = os.path.join(save_dir, 'metrics_denorm.txt')
        
        with open(denorm_file, 'w', encoding='utf-8') as f:
            f.write("Денормированные метрики по колонкам:\n")
            for name, value in denorm_metrics.items():
                f.write(f"{name}: {value:.6f}\n")


        df = pd.DataFrame({'idx': idxs.numpy()})
        for i, col in enumerate(self.config['target_cols']):
            df[f"{col}_true"] = targ_den[:, i].numpy()
            df[f"{col}_pred"] = preds_den[:, i].numpy()
        csv_path = os.path.join(save_dir, self.config.get('test_name', 'predictions.csv'))
        df.to_csv(csv_path, index=False)


        self.preds.clear()
        self.targets.clear()
        self.idxs.clear()