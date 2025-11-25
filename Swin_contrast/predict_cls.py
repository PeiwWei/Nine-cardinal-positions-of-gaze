import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from torchvision import transforms


from my_dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate, evaluate_cls

from sklearn import metrics
import copy


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    train_images_path, train_images_label, val_images_path, val_images_label, test_images_path, test_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "test": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    best_wts_path = './save_weights/multi500_BestAcc_0.6227272727272727.pkl'
    best_model_wts = torch.load(best_wts_path)
    model.load_state_dict(best_model_wts)
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    test_dataset = MyDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform=data_transform["test"])

    batch_size = 1
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=test_dataset.collate_fn)
    val_acc = evaluate_cls(model=model,
                            data_loader=val_loader,
                            device=device,
                            epoch=1, num_classes=args.num_classes)
    print('result_acc_val:', val_acc)

    test_acc = evaluate_cls(model=model,
                                 data_loader=test_loader,
                                 device=device,
                                 epoch=1, num_classes= args.num_classes)
    print('result_acc_test:', test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default="./data/multi")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    # ./vit_base_patch16_224_in21k.pth
    # 是否冻结权重
    # parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
