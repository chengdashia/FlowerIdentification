import os
from ultralytics import YOLO
from app.models.resnet import *
from app.models.resnet1 import *
from app.models.unet import Unet
import logging

logger = logging.getLogger(__name__)

def check_model_file(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if os.path.getsize(model_path) == 0:
        raise ValueError(f"模型文件为空: {model_path}")

def load_juhua_model(device=torch.device('cpu')):
    try:
        # 检查模型文件
        yolo_model_path = r'static/yolo_best.pt'
        model_path1 = r'static/ju_res101_huaxu.pth'
        model_path2 = r'static/ju_res101_huaxin.pth'
        
        for path in [yolo_model_path, model_path1, model_path2]:
            check_model_file(path)
        
        # 加载YOLO模型
        net1 = resnet().to(device)
        net1.load_state_dict(torch.load(model_path1, map_location=device))
        net1.eval()

        # 加载性状2模型
        net2 = resnet1().to(device)
        net2.load_state_dict(torch.load(model_path2, map_location=device))
        net2.eval()
        
        return YOLO(yolo_model_path), net1, net2
    except Exception as e:
        logger.error(f"加载菊花模型时出错: {str(e)}")
        raise

def load_filament_model():
    try:
        model_path = r'static/filament_model.pth'
        check_model_file(model_path)
        
        return Unet(
            model_path=model_path,
            num_classes=2,
            backbone="vgg",
            input_shape=[512, 512],
            mix_type=2,
            cuda=False  # 强制使用CPU
        )
    except Exception as e:
        logger.error(f"加载花丝模型时出错: {str(e)}")
        raise

def load_leaf_sheath_model():
    try:
        model_path = r'static/leaf_sheath_model.pth'
        check_model_file(model_path)
        
        return Unet(
            model_path=model_path,
            num_classes=2,
            backbone="vgg",
            input_shape=[512, 512],
            mix_type=2,
            cuda=False  # 强制使用CPU
        )
    except Exception as e:
        logger.error(f"加载叶鞘模型时出错: {str(e)}")
        raise
