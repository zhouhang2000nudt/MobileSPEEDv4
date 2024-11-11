from MobileSPEEDNetv4.model.MobileSPEEDv4 import MobileSPEEDNetv4
from MobileSPEEDNetv4.cfg import Config
import torch

from ptflops import get_model_complexity_info

config = Config()
model = MobileSPEEDNetv4(config)

input_shape = (1, 480, 768)

i = torch.randn(1, *input_shape)
o = model(i)

macs, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=True, flops_units="GMac", param_units="M")

print(f"模型 FLOPs: {macs}")
print(f"模型参数量: {params}")