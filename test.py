import torch
import torchvision
import cv2
import onnx
import numpy as np
import timm
import os
from PIL import Image
from torchvision import transforms
import onnxruntime
from onnxsim import simplify

print(torch.__version__)
print(cv2.__version__)
print(np.__version__)
print(onnx.__version__)


def init_model(model_name):
    if model_name == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
    if model_name == 'densnet':
        model = torchvision.models.densenet121(pretrained=True)
    if model_name == 'resnet':
        model = torchvision.models.resnet50(pretrained=True)
    if model_name == 'mobilenet':
        model = torchvision.models.mobilenet_v2(pretrained=True)
    if model_name == 'squeezenet':
        model = torchvision.models.squeezenet1_1(pretrained=True)
    if model_name == 'inception':
        model = torchvision.models.inception_v3(pretrained=False)
    if model_name == 'googlenet':
        model = torchvision.models.googlenet(pretrained=False)
    if model_name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=False)
    if model_name == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
    if model_name == 'shufflenet':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    if model_name == 'cspdarknet53':
        model = timm.create_model('cspdarknet53', pretrained=True)
    if model_name == 'seresnet18':
        model = timm.create_model('seresnet18', pretrained=True)
    if model_name == 'senet154':
        model = timm.create_model('senet154', pretrained=True)
    if model_name == 'seresnet50':
        model = timm.create_model('seresnet50', pretrained=True)
    if model_name == 'resnest50d':
        model = timm.create_model('resnest50d', pretrained=True)
    if model_name == 'skresnet50':
        model = timm.create_model('skresnet50', pretrained=True)
    model.eval()
    if model_name == 'inception':
        dummy = torch.randn(1, 3, 299, 299)
    else:
        dummy = torch.randn(1, 3, 224, 224)
    return model, dummy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, dummy = init_model('vgg16')
model = model.to(device)
dummy = dummy.to(device)

################################ pytorch转Onnx ######################################
onnx_name = 'vgg16.onnx'
torch.onnx.export(model, dummy, onnx_name)

# print("----- 模型测试 -----")
#
# def check_onnx_model(model, onnx_filename):
#     print(onnx_filename)
#     onnx_model = onnx.load(onnx_filename)
#     onnx.checker.check_model(onnx_model)
#     print("模型测试成功")
#     return onnx_model
#
# onnx_model = check_onnx_model(model, onnx_name)

# # -----  模型简化
# print("-----  模型简化 -----")
#
# # 输出模型名
# filename = onnx_name + "sim.onnx"
# # 简化模型
# # 设置skip_fuse_bn=True表示跳过融合bn层，pytorch高版本融合bn层会出错
# simplified_model, check = simplify(onnx_model, skip_fuse_bn=True)
# onnx.save_model(simplified_model, filename)
# onnx.checker.check_model(simplified_model)
#
# print("模型简化成功")

