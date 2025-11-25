import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from torchvision import transforms


from my_dataset import MyDataSet
from model.Swin.model import swin_tiny_patch4_window7_224 as create_model
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

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])


    test_dataset = MyDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform=data_transform["test"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

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

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc, train_acc_1, train_acc_2 = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        (val_loss, val_acc, val_acc_1, val_acc_2,
        sensitivity, specificity, precision, f1_score)= evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        if val_acc >= best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        tags = ["train_loss", "train_acc","train_acc_1","train_acc_2", "val_loss", "val_acc", "val_acc_1","val_acc_2",
                "sensitivity", "specificity", "precision", "f1_score",
                "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], train_acc_1, epoch)
        tb_writer.add_scalar(tags[3], train_acc_2, epoch)
        tb_writer.add_scalar(tags[4], val_loss, epoch)
        tb_writer.add_scalar(tags[5], val_acc, epoch)
        tb_writer.add_scalar(tags[6], val_acc_1, epoch)
        tb_writer.add_scalar(tags[7], val_acc_2, epoch)
        tb_writer.add_scalar(tags[8], sensitivity, epoch)
        tb_writer.add_scalar(tags[9], specificity, epoch)
        tb_writer.add_scalar(tags[10], precision, epoch)
        tb_writer.add_scalar(tags[11], f1_score, epoch)
        tb_writer.add_scalar(tags[12], optimizer.param_groups[0]["lr"], epoch)

        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
    model.load_state_dict(best_model_wts)
    print(best_acc)
    torch.save(model.state_dict(), "./save_weights/Vit_two" + str(args.epochs) + '_BestAcc_' + str(best_acc) + '.pkl')
    (test_loss, test_acc , test_acc_1, test_acc_2,
     sensitivity, specificity, precision, f1_score)= evaluate(model=model,
                                 data_loader=test_loader,
                                 device=device,
                                 epoch=epoch)
    print('result_acc_test: ', test_acc)
    print('result_acc_test: ', test_acc_1)
    print('result_acc_test: ', test_acc_2)
    print('result_sensitivity: ', sensitivity)
    print('result_specificity: ', specificity)
    print('result_precision: ', precision)
    print('result_f1_score: ', f1_score)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=4)
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
