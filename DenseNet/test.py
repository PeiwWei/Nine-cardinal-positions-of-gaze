import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from torchvision import transforms
from calflops import calculate_flops

from my_dataset import MyDataSet
# from vit_model import vit_base_patch16_224_in21k as create_model
from model.DenseNet.model import densenet121 as create_model
from utils import read_split_data, train_one_epoch, evaluate

from sklearn import metrics
import copy

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.Resize((192, 576)),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.2),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize((192, 576)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "test": transforms.Compose([transforms.Resize((192, 576)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集


    test_dataset = MyDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform=data_transform["test"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=test_dataset.collate_fn)

    model = create_model(num_classes=3).to(device)
    model_weight_path = "./weights/densenet.pkl"
    state = torch.load(model_weight_path, map_location=device)

    # new_key = 'encoder.layers.0.self_attn_long.self_attn_long.in_proj_weight'
    # value = state.pop('encoder.layers.0.self_attn_long.attn_long.in_proj_weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.0.self_attn_long.self_attn_long.in_proj_bias'
    # value = state.pop('encoder.layers.0.self_attn_long.attn_long.in_proj_bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.0.self_attn_long.self_attn_long.out_proj.weight'
    # value = state.pop('encoder.layers.0.self_attn_long.attn_long.out_proj.weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.0.self_attn_long.self_attn_long.out_proj.bias'
    # value = state.pop('encoder.layers.0.self_attn_long.attn_long.out_proj.bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.0.self_attn_short.self_attn_short.in_proj_weight'
    # value = state.pop('encoder.layers.0.self_attn_short.attn_short.in_proj_weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.0.self_attn_short.self_attn_short.in_proj_bias'
    # value = state.pop('encoder.layers.0.self_attn_short.attn_short.in_proj_bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.0.self_attn_short.self_attn_short.out_proj.weight'
    # value = state.pop('encoder.layers.0.self_attn_short.attn_short.out_proj.weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.0.self_attn_short.self_attn_short.out_proj.bias'
    # value = state.pop('encoder.layers.0.self_attn_short.attn_short.out_proj.bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.1.self_attn_long.self_attn_long.in_proj_weight'
    # value = state.pop('encoder.layers.1.self_attn_long.attn_long.in_proj_weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.1.self_attn_long.self_attn_long.in_proj_bias'
    # value = state.pop('encoder.layers.1.self_attn_long.attn_long.in_proj_bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.1.self_attn_long.self_attn_long.out_proj.weight'
    # value = state.pop('encoder.layers.1.self_attn_long.attn_long.out_proj.weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.1.self_attn_long.self_attn_long.out_proj.bias'
    # value = state.pop('encoder.layers.1.self_attn_long.attn_long.out_proj.bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.1.self_attn_short.self_attn_short.in_proj_weight'
    # value = state.pop('encoder.layers.1.self_attn_short.attn_short.in_proj_weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.1.self_attn_short.self_attn_short.in_proj_bias'
    # value = state.pop('encoder.layers.1.self_attn_short.attn_short.in_proj_bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.1.self_attn_short.self_attn_short.out_proj.weight'
    # value = state.pop('encoder.layers.1.self_attn_short.attn_short.out_proj.weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.1.self_attn_short.self_attn_short.out_proj.bias'
    # value = state.pop('encoder.layers.1.self_attn_short.attn_short.out_proj.bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.2.self_attn_long.self_attn_long.in_proj_weight'
    # value = state.pop('encoder.layers.2.self_attn_long.attn_long.in_proj_weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.2.self_attn_long.self_attn_long.in_proj_bias'
    # value = state.pop('encoder.layers.2.self_attn_long.attn_long.in_proj_bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.2.self_attn_long.self_attn_long.out_proj.weight'
    # value = state.pop('encoder.layers.2.self_attn_long.attn_long.out_proj.weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.2.self_attn_long.self_attn_long.out_proj.bias'
    # value = state.pop('encoder.layers.2.self_attn_long.attn_long.out_proj.bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.2.self_attn_short.self_attn_short.in_proj_weight'
    # value = state.pop('encoder.layers.2.self_attn_short.attn_short.in_proj_weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.2.self_attn_short.self_attn_short.in_proj_bias'
    # value = state.pop('encoder.layers.2.self_attn_short.attn_short.in_proj_bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.2.self_attn_short.self_attn_short.out_proj.weight'
    # value = state.pop('encoder.layers.2.self_attn_short.attn_short.out_proj.weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.2.self_attn_short.self_attn_short.out_proj.bias'
    # value = state.pop('encoder.layers.2.self_attn_short.attn_short.out_proj.bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.3.self_attn_long.self_attn_long.in_proj_weight'
    # value = state.pop('encoder.layers.3.self_attn_long.attn_long.in_proj_weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.3.self_attn_long.self_attn_long.in_proj_bias'
    # value = state.pop('encoder.layers.3.self_attn_long.attn_long.in_proj_bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.3.self_attn_long.self_attn_long.out_proj.weight'
    # value = state.pop('encoder.layers.3.self_attn_long.attn_long.out_proj.weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.3.self_attn_long.self_attn_long.out_proj.bias'
    # value = state.pop('encoder.layers.3.self_attn_long.attn_long.out_proj.bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.3.self_attn_short.self_attn_short.in_proj_weight'
    # value = state.pop('encoder.layers.3.self_attn_short.attn_short.in_proj_weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.3.self_attn_short.self_attn_short.in_proj_bias'
    # value = state.pop('encoder.layers.3.self_attn_short.attn_short.in_proj_bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.3.self_attn_short.self_attn_short.out_proj.weight'
    # value = state.pop('encoder.layers.3.self_attn_short.attn_short.out_proj.weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.3.self_attn_short.self_attn_short.out_proj.bias'
    # value = state.pop('encoder.layers.3.self_attn_short.attn_short.out_proj.bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.4.self_attn_long.self_attn_long.in_proj_weight'
    # value = state.pop('encoder.layers.4.self_attn_long.attn_long.in_proj_weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.4.self_attn_long.self_attn_long.in_proj_bias'
    # value = state.pop('encoder.layers.4.self_attn_long.attn_long.in_proj_bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.4.self_attn_long.self_attn_long.out_proj.weight'
    # value = state.pop('encoder.layers.4.self_attn_long.attn_long.out_proj.weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.4.self_attn_long.self_attn_long.out_proj.bias'
    # value = state.pop('encoder.layers.4.self_attn_long.attn_long.out_proj.bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.4.self_attn_short.self_attn_short.in_proj_weight'
    # value = state.pop('encoder.layers.4.self_attn_short.attn_short.in_proj_weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.4.self_attn_short.self_attn_short.in_proj_bias'
    # value = state.pop('encoder.layers.4.self_attn_short.attn_short.in_proj_bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.4.self_attn_short.self_attn_short.out_proj.weight'
    # value = state.pop('encoder.layers.4.self_attn_short.attn_short.out_proj.weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.4.self_attn_short.self_attn_short.out_proj.bias'
    # value = state.pop('encoder.layers.4.self_attn_short.attn_short.out_proj.bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.5.self_attn_long.self_attn_long.in_proj_weight'
    # value = state.pop('encoder.layers.5.self_attn_long.attn_long.in_proj_weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.5.self_attn_long.self_attn_long.in_proj_bias'
    # value = state.pop('encoder.layers.5.self_attn_long.attn_long.in_proj_bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.5.self_attn_long.self_attn_long.out_proj.weight'
    # value = state.pop('encoder.layers.5.self_attn_long.attn_long.out_proj.weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.5.self_attn_long.self_attn_long.out_proj.bias'
    # value = state.pop('encoder.layers.5.self_attn_long.attn_long.out_proj.bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.5.self_attn_short.self_attn_short.in_proj_weight'
    # value = state.pop('encoder.layers.5.self_attn_short.attn_short.in_proj_weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.5.self_attn_short.self_attn_short.in_proj_bias'
    # value = state.pop('encoder.layers.5.self_attn_short.attn_short.in_proj_bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.5.self_attn_short.self_attn_short.out_proj.weight'
    # value = state.pop('encoder.layers.5.self_attn_short.attn_short.out_proj.weight')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)
    # new_key = 'encoder.layers.5.self_attn_short.self_attn_short.out_proj.bias'
    # value = state.pop('encoder.layers.5.self_attn_short.attn_short.out_proj.bias')
    # new_key_value_pair = {new_key: value}
    # state.update(new_key_value_pair)

    model.load_state_dict(state)

    # batch_size = 1
    # input_shape = (batch_size, 3, 192, 576)
    # flops, macs, params = calculate_flops(model=model,
    #                                       input_shape=input_shape,
    #                                       output_as_string=True,
    #                                       output_precision=5)
    # print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))

    best_acc = 0.0
    print(best_acc)
    test_loss, test_acc , test_acc_1, test_acc_2 ,sensitivity, specificity, precision, f1_score= evaluate(model=model,
                                 data_loader=test_loader,
                                 device=device,
                                 epoch=0)
    print('result_acc_test: ', test_acc)
    print('result_acc_test_1: ', test_acc_1)
    print('result_acc_test_2: ', test_acc_2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default="./data")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # ./vit_base_patch16_224_in21k.pth
    # 是否冻结权重
    # parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
