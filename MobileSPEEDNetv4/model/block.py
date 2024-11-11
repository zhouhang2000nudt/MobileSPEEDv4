import torch
import torch.nn.functional as F
import torch.nn as nn

from timm.layers.mlp import Mlp, GluMlp

class MLP(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 act_layer: nn.Module = nn.SiLU,
                 bias: bool = True,
                 mlp_type: str = 'Mlp'):
        super().__init__()
        if mlp_type == 'Mlp':
            self.mlp = Mlp(in_features=in_features,
                           hidden_features=hidden_features,
                           out_features=out_features,
                           act_layer=act_layer,
                           bias=bias)
        elif mlp_type == 'GluMlp':
            self.mlp = GluMlp(in_features=in_features,
                              hidden_features=hidden_features*2,
                              out_features=out_features,
                              bias=bias)
        else:
            raise ValueError(f"mlp_type must be 'Mlp' or 'GluMlp', got {mlp_type}")
    
    def forward(self, x):
        return self.mlp(x)


class DeltaBranch(nn.Module):
    def __init__(self,
                 in_features_1: int,
                 in_features_2: int,
                 yaw_dim: int,
                 pitch_dim: int,
                 roll_dim: int,
                 pool_size_1: int,
                 pool_size_2: int,
                 act_layer: nn.Module = nn.SiLU,
                 bias: bool = True):
        super().__init__()
        mlp_infeaute_yaw = in_features_1 * pool_size_1**2 + in_features_2 * pool_size_2**2 + yaw_dim
        mlp_infeaute_pitch = in_features_1 * pool_size_1**2 + in_features_2 * pool_size_2**2 + pitch_dim
        mlp_infeaute_roll = in_features_1 * pool_size_1**2 + in_features_2 * pool_size_2**2 + roll_dim
        self.pool_size_1 = pool_size_1
        self.pool_size_2 = pool_size_2
        self.delta_mlp_yaw = MLP(in_features=mlp_infeaute_yaw,
                             hidden_features=mlp_infeaute_yaw // 2,
                             out_features=yaw_dim,
                             act_layer=act_layer,
                             bias=bias)
        self.delta_mlp_pitch = MLP(in_features=mlp_infeaute_pitch,
                                   hidden_features=mlp_infeaute_pitch // 2,
                                   out_features=pitch_dim,
                                   act_layer=act_layer,
                                   bias=bias)
        self.delta_mlp_roll = MLP(in_features=mlp_infeaute_roll,
                                  hidden_features=mlp_infeaute_roll // 2,
                                  out_features=roll_dim,
                                  act_layer=act_layer,
                                  bias=bias)
    
    def forward(self, x1, x2, yaw, pitch, row):
        x1 = F.adaptive_avg_pool2d(x1, self.pool_size_1).flatten(1) # B, C
        x2 = F.adaptive_avg_pool2d(x2, self.pool_size_2).flatten(1) # B, C
        delta_yaw = F.tanh(self.delta_mlp_yaw(torch.cat([x1, x2, yaw], dim=1)))
        delta_pitch = F.tanh(self.delta_mlp_pitch(torch.cat([x1, x2, pitch], dim=1)))
        delta_roll = F.tanh(self.delta_mlp_roll(torch.cat([x1, x2, row], dim=1)))
        delta_yaw = delta_yaw - torch.mean(delta_yaw, dim=1, keepdim=True)
        delta_pitch = delta_pitch - torch.mean(delta_pitch, dim=1, keepdim=True)
        delta_roll = delta_roll - torch.mean(delta_roll, dim=1, keepdim=True)
        return delta_yaw, delta_pitch, delta_roll