import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import cv2
import numpy
from model.ICSA.transformer import ICSA as create_model
from pytorch_grad_cam import GradCAM, \
                            ScoreCAM, \
                            GradCAMPlusPlus, \
                            AblationCAM, \
                            XGradCAM, \
                            EigenCAM, \
                            EigenGradCAM, \
                            LayerCAM, \
                            FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

output_cam_img_path1 = os.getcwd() + '/output_cam_img/output_cam_img2.jpg'
output_cam_img_path2 = os.getcwd() + '/output_cam_img/output_cam_img4.jpg'
raw_cam_img_path = os.getcwd() + "/raw_cam_img/raw_cam_img.jpg"
json_path = os.getcwd() + '/class_indices.json'
model_weight_path = os.getcwd() + "/weights/all_class.pkl"

def reshape_transform(tensor, height=36, width=12):
    # 去掉cls token
    print(tensor.shape)
    result = tensor[:, :, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def VIT_prodict (img_path):
    start_time = time.time()  # 记录程序开始时间
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()

    data_transform = transforms.Compose(
        [transforms.Resize((192, 576)),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]

    # 在OpenCV中，resize函数的语法格式为 宽在前，而高在后 , 在PyTorch中，transforms.Resize()函数的语法格式为高在前，而宽在后
    # 图像中显示的是宽在前，高在后
    rgb_img = cv2.resize(rgb_img, (576, 192))

    # 预处理图像
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # 看情况将图像转换为批量形式
    # input_tensor = input_tensor.unsqueeze(0)
    if use_cuda:
        input_tensor = input_tensor.cuda()

    # read class_indict

    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=3, has_logits=False).to(device)
    # load model weights

    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    # model.blocks[-1].norm1
    cam1 = XGradCAM(model=model,
                  target_layers=[model.encoder.layers[-1].norm2],
                  # 这里的target_layer要看模型情况，
                  # 比如还有可能是：target_layers = [model.blocks[-1].ffn.norm]
                  use_cuda=use_cuda,
                  reshape_transform=reshape_transform)
    target_category1 = None  # 可以指定一个类别，或者使用 None 表示最高概率的类别
    grayscale_cam1 = cam1(input_tensor=input_tensor, targets=target_category1)
    grayscale_cam1 = grayscale_cam1[0, :]

    # 将 grad-cam 的输出叠加到原始图像上
    # 转换图像类型为np.float32
    rgb_img1 = rgb_img.astype(numpy.float32)
    # 将像素值缩放到[0, 1]范围
    rgb_img1 /= 255.0

    visualization1 = show_cam_on_image(rgb_img1, grayscale_cam1)


    # 保存可视化结果
    cv2.cvtColor(visualization1, cv2.COLOR_RGB2BGR, visualization1)
    cv2.imwrite(output_cam_img_path1, visualization1)

    #####################################################################


    cam2 = XGradCAM(model=model,
                  target_layers=[model.encoder.layers[-1].norm4],
                  # 这里的target_layer要看模型情况，
                  # 比如还有可能是：target_layers = [model.blocks[-1].ffn.norm]
                  use_cuda=use_cuda,
                  reshape_transform=reshape_transform)
    target_category2 = None  # 可以指定一个类别，或者使用 None 表示最高概率的类别
    grayscale_cam2 = cam2(input_tensor=input_tensor, targets=target_category2)
    grayscale_cam2 = grayscale_cam2[0, :]




    # 将 grad-cam 的输出叠加到原始图像上
    # 转换图像类型为np.float32
    rgb_img2 = rgb_img.astype(numpy.float32)
    # 将像素值缩放到[0, 1]范围
    rgb_img2 /= 255.0

    visualization2 = show_cam_on_image(rgb_img2, grayscale_cam2)


    # 保存可视化结果
    cv2.cvtColor(visualization2, cv2.COLOR_RGB2BGR, visualization2)
    cv2.imwrite(output_cam_img_path2, visualization2)


    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))[0]).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    # plt.title(print_res)
    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))
    # plt.show()
    end_time = time.time()  # 记录程序结束时间
    all_time = end_time - start_time  # 计算程序使用时长（秒
    return all_time, class_indict, predict



if __name__ == '__main__':
    all_time, class_indict, predict = VIT_prodict(raw_cam_img_path)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
