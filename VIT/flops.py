import torch
from thop import profile
from vit_model import vit_base_patch16_224_in21k as create_model

from fvcore.nn import FlopCountAnalysis, parameter_count_table


def main():
    # 创建resnet50网络
    model = create_model()

    # 创建输入网络的tensor
    tensor = (torch.rand(1, 3, 192, 576),)

    # 分析FLOPs
    flops = FlopCountAnalysis(model, tensor)
    print("FLOPs: ", flops.total())

    # 分析parameters
    print(parameter_count_table(model))


if __name__ == '__main__':
    main()

