import comet_ml
import time
import os

import torch.autograd.gradcheck
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import argparse

from MobileSPEEDNetv4.cfg import Config
from MobileSPEEDNetv4.data import SpeedDataModule
from MobileSPEEDNetv4.model import LightningMobileSPEEDv4

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.plugins import MixedPrecision, DoublePrecision, Precision, BitsandbytesPrecision, HalfPrecision
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar, ModelSummary, RichModelSummary
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import DeviceStatsMonitor, LearningRateMonitor
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch import seed_everything


if __name__ == "__main__":
    
    # ====================配置====================
    config = Config()
    
    config.name = f"{config.name}-{config.stride}_{config.neighbor}_{config.alpha}"
    
    torch.set_float32_matmul_precision("high")
    
    dirpath = f"./result/{config.name}-{time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())}"
    # 判断是否存在路径 若不存在则创建
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    # 设置随机种子
    seed_everything(config.seed)

    # ===================训练器===================
    # =================callbacks=================
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # 进度条
    bar = RichProgressBar()
    # 保存模型
    checkpoint = ModelCheckpoint(dirpath=dirpath,
                                 filename="{epoch}-best",
                                 monitor="val/score",
                                 verbose=True,
                                 save_last=True,
                                 mode="min",
                                 save_weights_only=True)
    # 监控设备状态
    device_monitor = DeviceStatsMonitor(cpu_stats=None)
    # 模型总结
    summary = RichModelSummary(max_depth=3)
    callbacks = [lr_monitor, checkpoint, summary, bar]

    # ===================plugins=================
    plugins = []
    # 精度
    precision = MixedPrecision(precision="16-mixed", device=config.device)
    plugins = [precision]
    
    # ===================logger==================
    comet_logger = CometLogger(
        api_key=config.comet_api,
        save_dir=dirpath,
        project_name="MobileSPEEDv4",
        experiment_name=config.name + "-" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        offline=config.debug
    )
    
    # =================profiler==================
    profiler = SimpleProfiler(dirpath=dirpath)


    # ===================trainer=================
    if config.debug:
        config.num_workers = 1
        config.ram = False
        config.batch_size = 8
        config.epoch = 2
        config.offline = True
    else:
        torch.autograd.detect_anomaly(False)
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)
    trainer = Trainer(accelerator='gpu' if config.device == "cuda" else "cpu",        # 加速器
                      logger=comet_logger,
                      callbacks=callbacks,
                      max_epochs=config.epoch,
                      deterministic=config.deterministic,
                      benchmark=config.benchmark,
                    #   profiler=profiler,
                      plugins=plugins,
                    #   precision=precision,
                      default_root_dir=dirpath,
                      num_sanity_val_steps=0,
                      gradient_clip_val=2.0)

    # ====================模型====================
    # TODO Efficient initialization
    # with trainer.init_module():
    module = LightningMobileSPEEDv4(config)
    
    # ====================数据====================
    dataloader = SpeedDataModule(config)

    # ====================训练====================
    trainer.fit(model=module, datamodule=dataloader)

    # ====================验证====================
    # if config["val"]:
    #     module = LightningMobileSPEEDv4.load_from_checkpoint("/home/zh/pythonhub/yaolu/MobileSPEEDNetv4/result/epoch=254-best-95699.ckpt", config=config)
    #     trainer.validate(model=module, datamodule=dataloader)