from ultralytics import YOLO
from app.models.resnet import *
from app.models.resnet1 import *
from app.models.unet import Unet


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


def load_filament_model():
    device = torch.device('cpu')  # 强制使用CPU
    # 初始化模型（全局单例）
    unet_model = Unet(
        model_path=r'static/best_epoch_weights.pth',
        num_classes=2,
        backbone="vgg",
        input_shape=[512, 512],
        mix_type=2,
        cuda=False  # 强制使用CPU
    )
    return unet_model
