import torch
import rich

import lightning as L
import numpy as np

from torch.optim import AdamW

from ..cfg import Config
from .MobileSPEEDv4 import MobileSPEEDNetv4
from ..utils import OriLoss, PosLoss
from ..utils import Encoder, Decoder
from ..utils import Loss, PosError, OriError, Score

class LightningMobileSPEEDv4(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = MobileSPEEDNetv4(config)

        self.pos_loss = PosLoss(loss_type=config.pos_loss_type)
        self.yaw_loss = OriLoss(loss_type=config.ori_loss_type)
        self.pitch_loss = OriLoss(loss_type=config.ori_loss_type)
        self.roll_loss = OriLoss(loss_type=config.ori_loss_type)
        self.BETA = config.BETA

        self.decoder = Decoder(config.stride, config.alpha, config.neighbor, config.device)
        
        self.train_pos_loss = Loss()
        self.train_yaw_loss = Loss()
        self.train_pitch_loss = Loss()
        self.train_roll_loss = Loss()
        self.train_loss = Loss()
        self.val_pos_loss = Loss()
        self.val_yaw_loss = Loss()
        self.val_pitch_loss = Loss()
        self.val_roll_loss = Loss()
        self.val_loss = Loss()
        self.ori_error = OriError()
        self.pos_error = PosError()
        self.score = Score(config.ALPHA)
        
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.logger.experiment.log_asset_folder(folder="/home/zh/pythonhub/yaolu/MobileSPEEDv4/MobileSPEEDNetv4", log_file_name=True, recursive=True)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        num = inputs.shape[0]
        yaw, pitch, roll, pos = self.forward(inputs)
        train_pos_loss = self.pos_loss(pos, labels["pos"])
        train_yaw_loss = self.yaw_loss(yaw, labels["yaw_encode"])
        train_pitch_loss = self.pitch_loss(pitch, labels["pitch_encode"])
        train_roll_loss = self.roll_loss(roll, labels["roll_encode"])
        train_loss = self.BETA[0] * train_pos_loss + self.BETA[1] * (train_yaw_loss + train_pitch_loss + train_roll_loss)
        
        self.train_pos_loss.update(train_pos_loss.clone().detach(), num)
        self.train_yaw_loss.update(train_yaw_loss.clone().detach(), num)
        self.train_pitch_loss.update(train_pitch_loss.clone().detach(), num)
        self.train_roll_loss.update(train_roll_loss.clone().detach(), num)
        self.train_loss.update(train_loss.clone().detach(), num)
        
        return train_loss
    
    def on_train_epoch_end(self):
        self.log_dict({
            "train/pos_loss": self.train_pos_loss.compute(),
            "train/yaw_loss": self.train_yaw_loss.compute(),
            "train/pitch_loss": self.train_pitch_loss.compute(),
            "train/roll_loss": self.train_roll_loss.compute(),
            "train/loss": self.train_loss.compute()
        })
        self.train_pos_loss.reset()
        self.train_yaw_loss.reset()
        self.train_pitch_loss.reset()
        self.train_roll_loss.reset()
        self.train_loss.reset()
    
    def on_validation_start(self):
        rich.print(f"[b]{'train':<5} Epoch {self.current_epoch:>3}/{self.trainer.max_epochs:<3} pos_loss: {self.train_pos_loss.compute().item():<8.4f}  yaw_loss: {self.train_yaw_loss.compute().item():<8.4f}  pitch_loss: {self.train_pitch_loss.compute().item():<8.4f}  roll_loss: {self.train_roll_loss.compute().item():<8.4f}  loss: {self.train_loss.compute().item():<8.4f}")
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        num = inputs.shape[0]
        yaw, pitch, roll, pos = self.forward(inputs)
        val_pos_loss = self.pos_loss(pos, labels["pos"])
        val_yaw_loss = self.yaw_loss(yaw, labels["yaw_encode"])
        val_pitch_loss = self.pitch_loss(pitch, labels["pitch_encode"])
        val_roll_loss = self.roll_loss(roll, labels["roll_encode"])
        val_loss = self.BETA[0] * val_pos_loss + self.BETA[1] * (val_yaw_loss + val_pitch_loss + val_roll_loss)
        
        self.val_pos_loss.update(val_pos_loss.clone().detach(), num)
        self.val_yaw_loss.update(val_yaw_loss.clone().detach(), num)
        self.val_pitch_loss.update(val_pitch_loss.clone().detach(), num)
        self.val_roll_loss.update(val_roll_loss.clone().detach(), num)
        self.val_loss.update(val_loss.clone().detach(), num)
        
        ori_decode = self.decoder.decode_ori_batch(yaw, pitch, roll)
        self.ori_error.update(ori_decode, labels["ori"])
        self.pos_error.update(pos, labels["pos"])
    
    def on_validation_epoch_end(self):
        self.score.update(self.ori_error.compute(), self.pos_error.compute())
        self.log_dict({
            "val/pos_loss": self.val_pos_loss.compute(),
            "val/yaw_loss": self.val_yaw_loss.compute(),
            "val/pitch_loss": self.val_pitch_loss.compute(),
            "val/roll_loss": self.val_roll_loss.compute(),
            "val/loss": self.val_loss.compute(),
            "val/ori_error": self.ori_error.compute(),
            "val/pos_error": self.pos_error.compute(),
            "val/score": self.score.compute()
        })
        self.val_pos_loss.reset()
        self.val_yaw_loss.reset()
        self.val_pitch_loss.reset()
        self.val_roll_loss.reset()
        self.val_loss.reset()
        self.ori_error.reset()
        self.pos_error.reset()
        self.score.reset()
    
    def on_fit_end(self):
        self.logger.experiment.log_asset(self.trainer.callbacks[3].best_model_path, overwrite=True)
        self.logger.experiment.log_asset(self.trainer.callbacks[3].last_model_path, overwrite=True)
    
    def configure_optimizers(self):
        # 定义优化器
        optimizer = AdamW(self.parameters(), lr=self.config.lr0,
                            weight_decay=self.config.weight_decay)
        
        # 定义学习率调度器
        lr_scheduler_config: dict = {
            "scheduler": None,
            "interval": "epoch",            # 调度间隔
            "frequency": 1,                 # 调度频率
        }
        # 余弦退火学习率调度器
        lambda_max = self.config.lr0 / self.config.lr0
        lambda_min = self.config.lr_min / self.config.lr0
        warmup_epoch = self.config.warmup_epoch
        max_epoch = self.config.epoch
        lambda0 = lambda cur_iter: lambda_min + (lambda_max-lambda_min) * cur_iter / (warmup_epoch-1) if cur_iter < warmup_epoch \
            else lambda_min + (lambda_max-lambda_min)*(1 + np.cos(np.pi * (cur_iter - warmup_epoch) / (max_epoch - warmup_epoch - 1))) / 2
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        lr_scheduler_config["scheduler"] = scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }