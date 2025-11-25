import torch
from thop import profile
from model.ICSA.transformer import ICSA as create_model
from thop import profile
from torchstat import stat
from fvcore.nn import FlopCountAnalysis, parameter_count_table
model = create_model()
tensor = (torch.rand(1, 3, 192, 576),)
# def main():
#
#     # 分析FLOPs
#     flops = FlopCountAnalysis(model, tensor)
#     print("FLOPs: ", flops.total())
#
#     # 分析parameters
#     print(parameter_count_table(model))
#
#
# if __name__ == '__main__':
#     main()



# stat(model, ( 3, 192, 576))



# macs, params = profile(model, inputs=(tensor, ))
# print(' FLOPs: ', macs*2)   # 一般来讲，FLOPs是macs的两倍
# print('params: ', params)
print(sum(p.numel() for p in model.parameters()))
print(sum(p.numel() for p in model.parameters() if p.requires_grad))