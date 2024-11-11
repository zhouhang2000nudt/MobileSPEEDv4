import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from timm.layers.conv_bn_act import ConvBnAct

from .block import MLP, DeltaBranch
from ..cfg import Config

class MobileSPEEDNetv4(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.gray2rgb = ConvBnAct(in_channels=1,
                                  out_channels=3,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  act_layer=nn.Identity)
        self.backbone = timm.create_model('mobilenetv3_large_100', features_only=True)
        stages_channels = self.backbone.feature_info.channels()
        p3_channels, p4_channels, p5_channels = stages_channels[-3:]
        self.yaw_dim = 360 // config.stride + 1 + 2 * config.neighbor
        self.pitch_dim = 180 // config.stride + 1 + 2 * config.neighbor
        self.roll_dim = 360 // config.stride + 1 + 2 * config.neighbor
        self.pos_dim = config.pos_dim
        self.head = MLP(in_features=p5_channels,
                        hidden_features=p5_channels // 2,
                        out_features=self.yaw_dim + self.pitch_dim + self.roll_dim + self.pos_dim,
                        mlp_type=config.head_mlp_type)
        self.delta_branch = DeltaBranch(in_features_1=p3_channels,
                                        in_features_2=p4_channels,
                                        yaw_dim=self.yaw_dim,
                                        pitch_dim=self.pitch_dim,
                                        roll_dim=self.roll_dim,
                                        pool_size_1=config.pool_size[0],
                                        pool_size_2=config.pool_size[1])
    
    def forward(self, x):
        x = self.gray2rgb(x)
        o = self.backbone(x)
        last = self.head(F.adaptive_avg_pool2d(o[-1], 1).flatten(1))
        yaw, pitch, roll, pos = torch.split(last, [self.yaw_dim, self.pitch_dim, self.roll_dim, self.pos_dim], dim=1)
        yaw = F.softmax(yaw, dim=1)
        pitch = F.softmax(pitch, dim=1)
        roll = F.softmax(roll, dim=1)
        delta_yaw, delta_pitch, delta_roll = self.delta_branch(o[-3], o[-2], yaw, pitch, roll)
        yaw = yaw + delta_yaw
        pitch = pitch + delta_pitch
        roll = roll + delta_roll
        return yaw, pitch, roll, pos
        