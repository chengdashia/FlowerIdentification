from ultralytics import YOLO
from app.models.resnet import *
from app.models.resnet1 import *


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
