import timm
import torch
from rich import print
from calflops import calculate_flops

model_names = timm.list_models('*mobilenet*')
print(model_names)

model = timm.create_model('mobilenetv3_large_100', features_only=True)
print(model)
input_shape = (1, 3, 480, 768)
flops, macs, params = calculate_flops(model=model,
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
print(f"FLOPs: {flops} MACs: {macs} Params: {params}")
o = model(torch.randn(input_shape))
for x in o:
    print(x.shape)