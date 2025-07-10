import os
from ultralytics import YOLO
from app.models.ym.ultralytics import YOLO as YMYOLO
from app.models.resnet import *
from app.models.resnet1 import *
from app.models.unet import Unet
from app.models.ym.YMUNet import YMUNet
import logging

logger = logging.getLogger(__name__)


def check_model_file(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if os.path.getsize(model_path) == 0:
        raise ValueError(f"模型文件为空: {model_path}")


def load_chr_model(device=torch.device('cpu')):
    try:
        # 检查模型文件
        yolo_model_path = r'static/models/chr/yolo_best.pt'
        model_path1 = r'static/models/chr/ju_res101_huaxu.pth'
        model_path2 = r'static/models/chr/ju_res101_huaxin.pth'
        
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
        model_path = r'static/models/filament/filament_model.pth'
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
        model_path = r'static/models/leaf/leaf_sheath_model.pth'
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


def load_ym_models():
    """加载YM分析所需的模型"""
    try:
        # 配置路径
        yolo_weights = r'static/models/ym2/best.pt'
        unet_weights = r'static/models/ym2/best_unet_model.pth'
        zl_weights = r'static/models/ym2/YMCK-best.pt'

        # 加载YOLO模型
        yolo_model = YMYOLO(yolo_weights)
        zl_model = YMYOLO(zl_weights)

        # 加载U-Net模型
        device = torch.device('cpu')
        unet_model = YMUNet(n_channels=3, n_classes=1, n_filters=32)
        unet_model.load_state_dict(torch.load(unet_weights, map_location=device))
        unet_model.to(device)
        unet_model.eval()

        # ym-unet
        ym_unet = Unet(
            model_path=r'static/models/ym2/best_epoch_weights.pth',
            num_classes=3,
            backbone="vgg",
            input_shape=[512, 512],
            mix_type=0,
        )

        return yolo_model, unet_model, zl_model, ym_unet

    except Exception as e:
        print(f"加载YM模型失败: {e}")
        raise e


def load_ym2_models():
    """加载YM分析所需的模型"""
    try:
        # 配置路径
        yolo_weights = r"static\models\ym2\best.pt"
        unet_weights = r"static\models\ym2\best_unet_model.pth"
        model_path = r"static\weights\ym2\YMCK-best.pt"

        # 加载YOLO模型
        yolo_model = YMYOLO(yolo_weights)

        # 加载U-Net模型
        device = torch.device('cpu')
        unet_model = YMUNet(n_channels=3, n_classes=1, n_filters=32)
        unet_model.load_state_dict(torch.load(unet_weights, map_location=device))
        unet_model.to(device)
        unet_model.eval()

        return yolo_model, unet_model

    except Exception as e:
        print(f"加载YM模型失败: {e}")
        raise e


