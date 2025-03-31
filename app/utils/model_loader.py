import torch
from ultralytics import YOLO
import app.models.model_bicnn as model_bicnn
import app.models.model_bicnn2 as model_bicnn2
from app.models.resnet import *
from app.models.resnet1 import *

def load_models(device=torch.device('cpu')):
    # 加载 BICNN 模型1
    model_path1 = 'static/bcnn_alllayer1.pth'
    net1 = model_bicnn.Net().to(device)
    net1.load_state_dict(torch.load(model_path1, map_location=device))
    net1.eval()

    # 加载 BICNN 模型2
    model_path2 = 'static/bcnn_alllayer2.pth'
    net2 = model_bicnn2.Net().to(device)
    net2.load_state_dict(torch.load(model_path2, map_location=device))
    net2.eval()

    # 加载 YOLO 模型
    yolo_model_path = r'static/best.pt'
    yolo_model = YOLO(yolo_model_path)

    return net1, net2, yolo_model


def load_juhua_model(device=torch.device('cpu')):
    # 加载YOLO模型
    yolo_model_path = r'static/yolo_best.pt'
    model_path1 = r'static/ju_res101_huaxu.pth'
    net1 = resnet().to(device)
    net1.load_state_dict(torch.load(model_path1, map_location=device))
    net1.eval()

    # 加载性状2模型
    model_path2 = r'static/ju_res101_huaxin.pth'
    net2 = resnet1().to(device)
    net2.load_state_dict(torch.load(model_path2, map_location=device))
    net2.eval()
    return YOLO(yolo_model_path), net1, net2
