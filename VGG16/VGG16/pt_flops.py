import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
from model.ICSA.transformer import ICSA as create_model


with torch.cuda.device(0):
  device = torch.device(0)
  model = create_model(num_classes=10, has_logits=False).to(device)
  macs, params = get_model_complexity_info(model, (3, 192, 576), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


#
# import torch
# import torch.nn as nn
#
# # 计算GCNConv运算量的函数
# def calc_gcnconv_flops(weight, adj):
#     # 获取输入形状和参数数量
#     input_size = int(adj.shape[0])
#     output_size = int(weight.shape[1])
#
#     # 计算浮点计算量
#     flops = (2 * input_size * output_size)**2
#
#     # 加上参数更新的计算量
#     weight_count = weight.numel()
#     flops += 2 * input_size * output_size * weight_count
#
#     return flops
#
# # 创建一个包含GCNConv层的模型
# model = nn.Sequential(
#     nn.Linear(16, 32),
#     nn.ReLU(),
#     nn.Linear(32, 64),
#     nn.ReLU(),
#     nn.Linear(64, 128),
#     nn.ReLU()
# )
#
# # 定义输入张量和邻接矩阵
# input_tensor = torch.randn((1, 16))
# adj = torch.randn((16, 16))
#
# # 遍历模型，计算每一层的FLOPS并累加
# total_flops = 0
# for layer in model.children():
#     if isinstance(layer, nn.Linear):
#         flops = calc_gcnconv_flops(layer.weight, adj)
#         total_flops += flops
#     input_tensor = layer(input_tensor)
#
# print("Total FLOPS: {:.2f} GFLOPS".format(total_flops / 1e9))
