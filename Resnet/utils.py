import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from builtins import print
import pandas as pd
from xpinyin import Pinyin
import os
import numpy
import shutil
from sklearn.metrics import confusion_matrix, roc_curve, auc

def read_split_data(root: str, val_rate: float = 0.3, test_rate: float = 0.1):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    eye_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    eye_class.sort()
    # 生成类别名称以及对应的数字索引
    # class_indices = dict((k, v) for v, k in enumerate(eye_class))

    '''
    固定下来 class_indices 字典方便对应
    
    '''
    class_indices = {'dvd_no': [0, 0], 'eso_A': [1, 1], 'eso_no': [1, 0],
                     'eso_V': [1, 2], 'exo_A': [2, 1], 'exo_no': [2, 0],
                     'exo_V': [2, 2]}

    # json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    # with open('class_indices.json', 'w') as json_file:
    #     json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    test_images_path = []  # 存储测试集的所有图片路径
    test_images_label = []  # 存储测试集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件

    ### 这一段代码是要每类文件夹下分train、val、test的（结束） ###
    for cla in eye_class:
        cla_path = os.path.join(root, cla)
        cla_path_train = os.path.join(cla_path, 'train')
        print(cla_path_train)
        cla_path_val = os.path.join(cla_path, 'val')
        cla_path_test = os.path.join(cla_path, 'test')
        # 遍历获取supported支持的所有文件路径
        train_img_path = [os.path.join(cla_path_train, i) for i in os.listdir(cla_path_train) if os.path.splitext(i)[-1] in supported]
        val_img_path = [os.path.join(cla_path_val, i) for i in os.listdir(cla_path_val) if os.path.splitext(i)[-1] in supported]
        test_img_path = [os.path.join(cla_path_test, i) for i in os.listdir(cla_path_test) if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(train_img_path) + len(val_img_path) + len(test_img_path))

        for img_path in train_img_path:
            train_images_path.append(img_path)
            train_images_label.append(image_class)
        for img_path in val_img_path:
            val_images_path.append(img_path)
            val_images_label.append(image_class)
        for img_path in test_img_path:
            test_images_path.append(img_path)
            test_images_label.append(image_class)
    ### 这一段代码是要每类文件夹下分train、val、test的（结束） ###

    ### 这一段代码是不用每类文件夹下分train、val、test的 ###
    # for cla in eye_class:
    #     cla_path = os.path.join(root, cla)
    #     # 遍历获取supported支持的所有文件路径
    #     images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
    #               if os.path.splitext(i)[-1] in supported]
    #     # 获取该类别对应的索引
    #     image_class = class_indices[cla]
    #     # 记录该类别的样本数量
    #     every_class_num.append(len(images))
    #     # # 按比例随机采样验证样本
    #     # val_path = random.sample(images, k=int(len(images) * val_rate))
    #     ld1 = int(len(images) * (1- val_rate -test_rate))
    #     ld2 = int(len(images) * (1 -test_rate))
    #     train_path = images[:ld1]
    #     val_path = images[ld1:ld2]
    #     test_path = images[ld2:]
    #
    #     for img_path in images:
    #         if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
    #             val_images_path.append(img_path)
    #             val_images_label.append(image_class)
    #         elif img_path in train_path:  # 否则存入训练集
    #             train_images_path.append(img_path)
    #             train_images_label.append(image_class)
    #         else:  # 否则存入训练集
    #             test_images_path.append(img_path)
    #             test_images_label.append(image_class)
    ### 这一段代码是不用每类文件夹下分train、val、test的（结束） ###

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    print("{} images for test.".format(len(test_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(eye_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(eye_class)), eye_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label , test_images_path, test_images_label


def read_split_multi_data(root: str, val_rate: float = 0.3, test_rate: float = 0.1):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    eye_class = ['DVD', 'EXSO', 'AV']
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(eye_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    label_excel_path = root + '/all_data_label.xlsx'
    df = pd.read_excel(label_excel_path).copy()


    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件

    cla_path_train = os.path.join(root, 'train')
    cla_path_val = os.path.join(root, 'val')
    cla_path_test = os.path.join(root, 'test')
    # 遍历获取supported支持的所有文件路径
    train_img_path = [os.path.join(cla_path_train, i) for i in os.listdir(cla_path_train) if os.path.splitext(i)[-1] in supported]
    val_img_path = [os.path.join(cla_path_val, i) for i in os.listdir(cla_path_val) if os.path.splitext(i)[-1] in supported]
    test_img_path = [os.path.join(cla_path_test, i) for i in os.listdir(cla_path_test) if os.path.splitext(i)[-1] in supported]
    train_images_label = [[] for _ in range(len(train_img_path))]
    val_images_label = [[] for _ in range(len(val_img_path))]
    test_images_label = [[] for _ in range(len(test_img_path))]
    for n in range(0, int(len(df))):

        row_n = df.iloc[n]
        name = row_n['姓名']
        time = row_n['时间']
        new =  str(time) + name + '.jpg'
        new_train_path = os.path.join(cla_path_train, new)
        new_val_path = os.path.join(cla_path_val, new)
        new_test_path = os.path.join(cla_path_test, new)
        label = []
        if row_n['DVD'] == 1:
            label.append(1)
        else :
            label.append(0)
        if row_n['内斜视'] != 0 or row_n['外斜视'] != 0:
            label.append(1)
        else :
            label.append(0)
        if row_n['A征'] != 0 or row_n['V征'] != 0:
            label.append(1)
        else :
            label.append(0)
        if new_train_path in train_img_path:
            index = train_img_path.index(new_train_path)
            train_images_label[index].append(label)
        elif new_val_path in val_img_path:
            index = val_img_path.index(new_val_path)
            val_images_label[index].append(label)
        elif new_test_path in test_img_path:
            index = test_img_path.index(new_test_path)
            test_images_label[index].append(label)
    train_images_label = numpy.squeeze(train_images_label)
    val_images_label = numpy.squeeze(val_images_label)
    test_images_label = numpy.squeeze(test_images_label)
    print(train_images_label)
    return train_img_path, train_images_label, val_img_path, val_images_label, test_img_path, test_images_label



def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def calculate_metrics(pred_result, labels):
    pred_labels = torch.argmax(pred_result, dim=1)

    # 计算精确度（Accuracy）
    accuracy = torch.sum(pred_labels == labels).item() / len(labels)

    # 计算灵敏度（Sensitivity/Recall）
    cm = confusion_matrix(labels, pred_labels)
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    # 计算特异度（Specificity）
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    # 计算精确率（Precision）
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])

    # 计算F1值（F1-score）
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    # 计算AUC值（Area Under the ROC Curve）
    fpr, tpr, thresholds = roc_curve(labels, pred_labels)
    auc_value = auc(fpr, tpr)

    return accuracy, sensitivity, specificity, precision, f1_score, auc_value


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num_1 = torch.zeros(1).to(device)   # 第一层累计预测正确的样本数
    accu_num_2 = torch.zeros(1).to(device)  # 第二层累计预测正确的样本数
    accu_num_all = torch.zeros(1).to(device)  # 总计累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        labels_1 = labels[:, 0]  # 第一列为label1
        labels_2 = labels[:, 1]  # 第二列为label2
        sample_num += images.shape[0]

        pred_1, pred_2 = model(images.to(device))
        pred_classes_1 = torch.max(pred_1, dim=1)[1]
        pred_classes_2 = torch.max(pred_2, dim=1)[1]
        accu_num_1 += torch.eq(pred_classes_1, labels_1.to(device)).sum()
        accu_num_2 += torch.eq(pred_classes_2, labels_2.to(device)).sum()

        for i in range(len(pred_classes_1)):
            if pred_classes_1[i] == labels_1[i].to(device) and pred_classes_2[i] == labels_2[i].to(device):
                accu_num_all += 1
        loss_1 = loss_function(pred_1, labels_1.to(device))
        loss_2 = loss_function(pred_2, labels_2.to(device))
        loss = loss_1 + loss_2
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, acc_1: {:.3f}, acc_2: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num_all.item() / sample_num,
                                                                               accu_num_1.item() / sample_num,
                                                                               accu_num_2.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num_all.item() / sample_num, accu_num_1.item() / sample_num, accu_num_2.item() / sample_num

def multi_train_one_epoch(model, optimizer, data_loader, device, epoch):
    pred_thea = 0.5
    model.train()
    loss_function = torch.nn.MultiLabelSoftMarginLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_DVD = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_EXSO = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_AV = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        # print(pred)
        pred_classes = torch.sigmoid(pred)
        # print(pred_classes , labels.to(device) )
        zero = torch.zeros_like(pred_classes)
        one = torch.ones_like(pred_classes)
        # a中大于0.5的用one(1)替换,否则a替换,即不变
        pred_result = torch.where(pred_classes > pred_thea, one, pred_classes)

        # a中小于0.5的用zero(0)替换,否则a替换,即不变
        pred_result = torch.where(pred_result < pred_thea, zero, pred_result)
        accu_num += torch.eq(pred_result.to(device), labels.to(device)).sum()

        accu_DVD += torch.eq(pred_result.to(device)[:, 0], labels.to(device)[:, 0]).sum()
        accu_EXSO += torch.eq(pred_result.to(device)[:, 1], labels.to(device)[:, 1]).sum()
        accu_AV += torch.eq(pred_result.to(device)[:, 2], labels.to(device)[:, 2]).sum()


        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc_all: {:.3f}, acc_DVD: {:.3f}, acc_EXSO: {:.3f}, acc_AV: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / (sample_num *3), accu_DVD.item() / sample_num,
                                                                                accu_EXSO.item() / sample_num,  accu_AV.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / (sample_num *3) , accu_DVD.item() / sample_num, accu_EXSO.item() / sample_num,  accu_AV.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num_1 = torch.zeros(1).to(device)   # 第一层累计预测正确的样本数
    accu_num_2 = torch.zeros(1).to(device)  # 第二层累计预测正确的样本数
    accu_num_all = torch.zeros(1).to(device)  # 总计累计预测正确的样本数
    true_labels_1 = []
    pred_labels_1 = []
    true_labels_2 = []
    pred_labels_2 = []

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data

        labels_1 = labels[:, 0]  # 第一列为label1
        labels_2 = labels[:, 1]  # 第二列为label2
        sample_num += images.shape[0]

        pred_1, pred_2 = model(images.to(device))

        pred_classes_1 = torch.max(pred_1, dim=1)[1]
        pred_classes_2 = torch.max(pred_2, dim=1)[1]
        accu_num_1 += torch.eq(pred_classes_1, labels_1.to(device)).sum()
        accu_num_2 += torch.eq(pred_classes_2, labels_2.to(device)).sum()

        for i in range(len(pred_classes_1)):
            if pred_classes_1[i] == labels_1[i].to(device) and pred_classes_2[i] == labels_2[i].to(device):
                accu_num_all += 1

        true_labels_1.extend(labels_1.cpu().detach().numpy())
        true_labels_2.extend(labels_2.cpu().detach().numpy())
        pred_labels_1.extend(pred_classes_1.cpu().detach().numpy())
        pred_labels_2.extend(pred_classes_2.cpu().detach().numpy())

        loss_1 = loss_function(pred_1, labels_1.to(device))
        loss_2 = loss_function(pred_2, labels_2.to(device))
        loss = loss_1 + loss_2
        accu_loss += loss


        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}, acc_1: {:.3f}, acc_2: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num_all.item() / sample_num,
                                                                               accu_num_1.item() / sample_num,
                                                                               accu_num_2.item() / sample_num)
    cm_1 = confusion_matrix(true_labels_1, pred_labels_1)
    # print("cm_1")
    # print(cm_1)
    cm_2 = confusion_matrix(true_labels_2, pred_labels_2)

    # 下面的四个指标都是使用微平均统计下来的结果

    TP = (cm_1[0, 0] + cm_1[1, 1] + cm_1[2, 2] + cm_2[0, 0] + cm_2[1, 1] + cm_2[2, 2])
    TN = (cm_1[0, 0] + cm_1[1, 1] + cm_1[2, 2] + cm_2[0, 0] + cm_2[1, 1] + cm_2[2, 2]) * 2 # 非该类以外预测正确的
    FP = (cm_1[0, 1] + cm_1[0, 2] + cm_1[1, 0] + cm_1[1, 2] + cm_1[2, 0] + cm_1[2, 1] +
          cm_2[0, 1] + cm_2[0, 2] + cm_2[1, 0] + cm_2[1, 2] + cm_2[2, 0] + cm_2[2, 1])
    FN = (cm_1[0, 1] + cm_1[0, 2] + cm_1[1, 0] + cm_1[1, 2] + cm_1[2, 0] + cm_1[2, 1] +
          cm_2[0, 1] + cm_2[0, 2] + cm_2[1, 0] + cm_2[1, 2] + cm_2[2, 0] + cm_2[2, 1]) * 2
    # 计算灵敏度（Sensitivity/Recall）TP / TP + FN ： 预测为A正确的 + 预测为V正确的 + 预测为非AV正确的 / A征label数 + V征label数 + 非AV征数
    sensitivity = TP / (TP + FN)

    # 计算特异度（Specificity）TN / FP +TN
    specificity = TN / (TN + FP)

    # 计算精确率（Precision） TP / TP + FP
    precision = TP / (TP + FP)

    # 计算F1值（F1-score）
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    print('epoch: ', epoch)

    print('result_precision: ', precision)
    print('result_sensitivity: ', sensitivity)
    print('result_specificity: ', specificity)

    print('result_f1_score: ', f1_score)
    return (accu_loss.item() / (step + 1), accu_num_all.item() / sample_num, accu_num_1.item() / sample_num, accu_num_2.item() / sample_num,
            sensitivity, specificity, precision, f1_score)



@torch.no_grad()
def multi_evaluate(model, data_loader, device, epoch):
    pred_thea = 0.5
    loss_function = torch.nn.MultiLabelSoftMarginLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_DVD = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_EXSO = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_AV = torch.zeros(1).to(device)  # 累计预测正确的样本数

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.sigmoid(pred)
        zero = torch.zeros_like(pred_classes)
        one = torch.ones_like(pred_classes)
        pred_result = torch.where(pred_classes > pred_thea, one, pred_classes)
        pred_result = torch.where(pred_result < pred_thea, zero, pred_result)
        accu_num += torch.eq(pred_result.to(device), labels.to(device)).sum()
        accu_DVD += torch.eq(pred_result.to(device)[:, 0], labels.to(device)[:, 0]).sum()
        accu_EXSO += torch.eq(pred_result.to(device)[:, 1], labels.to(device)[:, 1]).sum()
        accu_AV += torch.eq(pred_result.to(device)[:, 2], labels.to(device)[:, 2]).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc_all: {:.3f}, acc_DVD: {:.3f}, acc_EXSO: {:.3f}, acc_AV: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / (sample_num *3), accu_DVD.item() / sample_num,
            accu_EXSO.item() / sample_num, accu_AV.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / (sample_num *3) , accu_DVD.item() / sample_num, accu_EXSO.item() / sample_num,  accu_AV.item() / sample_num



@torch.no_grad()
def evaluate_cls(model, data_loader, device, epoch, num_classes):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    cls_result = torch.zeros(num_classes).to(device)
    num_result = torch.zeros(num_classes).to(device)
    final_result = torch.zeros(num_classes).to(device)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        cls_result[int(labels.to(device))] += torch.eq(pred_classes, labels.to(device)).sum()
        for i in range(0, num_classes):
            num_result[i] += torch.eq(torch.tensor(i), labels.to(device)).sum()

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

    for i in range(0, num_classes):
        final_result[i] = cls_result[i] / num_result[i]
        print(cls_result[i], num_result[i])
    return final_result