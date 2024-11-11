import torch

import numpy as np

from scipy.spatial.transform import Rotation as R

class EDCoder:
    init = False
    
    def __init__(self, stride: int, alpha: float, neighbor: int, device: str = "cpu"):
        EDCoder.init = True
        EDCoder.stride = stride
        EDCoder.alpha = alpha
        EDCoder.neighbor = neighbor
        EDCoder.device = device
        EDCoder.yaw_len = int(360 // stride + 1 + 2*neighbor)
        EDCoder.pitch_len = int(180 // stride + 1 + 2*neighbor)
        EDCoder.roll_len = int(360 // stride + 1 + 2*neighbor)
        EDCoder.yaw_range = torch.linspace(-neighbor * stride, 360 + neighbor * stride, EDCoder.yaw_len, device=device) - 180        # -180 ~ 180
        EDCoder.pitch_range = torch.linspace(-neighbor * stride, 180 + neighbor * stride, EDCoder.pitch_len, device=device) - 90        # -90 ~ 90
        EDCoder.roll_range = torch.linspace(-neighbor * stride, 360 + neighbor * stride, EDCoder.roll_len, device=device) - 180        # -180 ~ 180
        EDCoder.yaw_index_dict = {int(yaw // stride): i for i, yaw in enumerate(EDCoder.yaw_range)}
        EDCoder.pitch_index_dict = {int(pitch // stride): i for i, pitch in enumerate(EDCoder.pitch_range)}
        EDCoder.roll_index_dict = {int(roll // stride): i for i, roll in enumerate(EDCoder.roll_range)}

class Encoder(EDCoder):
    def __init__(self, stride: int, alpha: float, neighbor: int, device: str = "cpu"):
        if not EDCoder.init:
            super().__init__(stride, alpha, neighbor, device)
    
    def _encode_ori(self, angle: float, angle_len: int, index_dict: dict):
        angle_encode = np.zeros(angle_len, dtype=np.float32)
        
        mean = angle / self.stride
        l, r = int(np.floor(mean)), int(np.ceil(mean))
        if l == r:
            angle_encode[index_dict[l]] = 1
        else:
            angle_encode[index_dict[l]] = (r - mean) / (r - l)
            angle_encode[index_dict[r]] = (mean - l) / (r - l)
        weight = 1
        for _ in range(self.neighbor):
            angle_encode[index_dict[l]] *= (1 - self.alpha)
            angle_encode[index_dict[r]] *= (1 - self.alpha)
            l -= 1
            r += 1
            weight *= self.alpha
            angle_encode[index_dict[l]] = (r - mean) / (r - l) * weight
            angle_encode[index_dict[r]] = (mean - l) / (r - l) * weight
        
        return angle_encode
    
    def encode_ori(self, ori: np.ndarray):
        rotation = R.from_quat([ori[1], ori[2], ori[3], ori[0]])
        yaw, pitch, roll = rotation.as_euler("YXZ", degrees=True)
        
        yaw_encode = self._encode_ori(yaw, self.yaw_len, self.yaw_index_dict)
        pitch_encode = self._encode_ori(pitch, self.pitch_len, self.pitch_index_dict)
        roll_encode = self._encode_ori(roll, self.roll_len, self.roll_index_dict)
        
        return yaw_encode, pitch_encode, roll_encode

class Decoder(EDCoder):
    def __init__(self, stride: int, alpha: float, neighbor: int, device: str = "cpu"):
        if not EDCoder.init:
            super().__init__(stride, alpha, neighbor, device)
    
    def _decode_ori(self, encode: torch.Tensor, angle_range: torch.Tensor):
        return torch.sum(encode * angle_range, dim=1)
    
    def decode_ori(self, yaw_encode: torch.Tensor, pitch_encode: torch.Tensor, roll_encode: torch.Tensor):
        yaw = self._decode_ori(yaw_encode, self.yaw_range)
        pitch = self._decode_ori(pitch_encode, self.pitch_range)
        roll = self._decode_ori(roll_encode, self.roll_range)
        
        rotation = R.from_euler('YXZ', [yaw, pitch, roll], degrees=True)
        ori = rotation.as_quat()
        ori = [ori[3], ori[0], ori[1], ori[2]]
        
        return torch.tensor(ori)
    
    def decode_ori_batch(self, yaw_encode, pitch_encode, roll_encode):
        
        yaw_decode = torch.sum(yaw_encode * self.yaw_range, dim=1)
        pitch_decode = torch.sum(pitch_encode * self.pitch_range, dim=1)
        roll_decode = torch.sum(roll_encode * self.roll_range, dim=1)
        
        cy = torch.cos(torch.deg2rad(yaw_decode * 0.5))
        sy = torch.sin(torch.deg2rad(yaw_decode * 0.5))
        cp = torch.cos(torch.deg2rad(pitch_decode * 0.5))
        sp = torch.sin(torch.deg2rad(pitch_decode * 0.5))
        cr = torch.cos(torch.deg2rad(roll_decode * 0.5))
        sr = torch.sin(torch.deg2rad(roll_decode * 0.5))
        
        q0 = cy * cp * cr + sy * sp * sr
        q1 = cy * sp * cr + sy * cp * sr
        q2 = sy * cp * cr - cy * sp * sr
        q3 = -sy * sp * cr + cy * cp * sr
        
        return torch.stack([q0, q1, q2, q3], dim=1)