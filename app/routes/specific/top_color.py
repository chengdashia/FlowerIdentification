import os
import torch
from PIL import Image
import cv2
import numpy as np
from skimage import color

from app import api
from flask_restx import Namespace, Resource
from flask import request, jsonify, Blueprint, make_response
import logging
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

# 导入您的模型
from app.models.corn.ultralytics import YOLO
from app.models.corn.specific.rachis.unet import Unet
from app.utils.path_utils import convert_to_url_path

logger = logging.getLogger(__name__)

# 设置环境变量以解决OpenMP运行时库重复加载的问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 创建蓝图和命名空间
corn_top_color_bp = Blueprint('corn_top_color', __name__)
corn_top_color_ns = Namespace('corn_top_color', description='玉米果穗顶端颜色分析API')

# 获取应用根目录的绝对路径
app_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# 配置上传目录
UPLOAD_FOLDER = os.path.join(app_root, 'static', 'images/corn_top_color/uploads')
RESULT_FOLDER = os.path.join(app_root, 'static', 'images/corn_top_color/results')
CROP_FOLDER = os.path.join(app_root, 'static', 'static/images/corn_top_color/crops')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# 模型路径配置
# YOLO_WEIGHT_PATH = r'static/models/specific/topColor/best.pt'
# UNET_WEIGHT_PATH = r'static/models/specific/topColor/best_epoch_weights.pth'

YOLO_WEIGHT_PATH = os.path.join(app_root, 'static/models/specific/topColor/best.pt')
UNET_WEIGHT_PATH = os.path.join(app_root, 'static/models/specific/topColor/best_epoch_weights.pth')

# UNet模型参数
NUM_CLASSES = 2
BACKBONE = "vgg"
INPUT_SHAPE = [512, 512]
MIX_TYPE = 0

# 类别索引
GS_CLASS_INDEX = 1  # 果穗类别

# 确保目录存在
for folder in [UPLOAD_FOLDER, CROP_FOLDER, RESULT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 全局模型缓存
_MODEL_CACHE = None


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------- 核心分析类 --------------------- #
class CornTopColorAnalyzer:
    """玉米果穗顶端颜色分析器"""

    def __init__(self, yolo_weight_path, unet_weight_path,
                 num_classes=3, backbone="vgg", input_shape=[512, 512], mix_type=0):
        """
        初始化分析器

        Args:
            yolo_weight_path: YOLO模型权重路径
            unet_weight_path: UNet模型权重路径
            num_classes: UNet类别数量
            backbone: UNet骨干网络
            input_shape: UNet输入图像大小
            mix_type: UNet混合类型
        """
        if not os.path.exists(yolo_weight_path):
            raise FileNotFoundError(f"YOLO 权重不存在: {yolo_weight_path}")

        if not os.path.exists(unet_weight_path):
            raise FileNotFoundError(f"UNet 权重不存在: {unet_weight_path}")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 加载YOLO模型
        try:
            self.yolo_model = YOLO(yolo_weight_path)
            logger.info(f"YOLO 加载完成 | 设备: {self.device}")
        except Exception as e:
            raise ValueError(f"YOLO模型加载失败: {str(e)}")

        # 加载UNet模型
        try:
            self.unet_model = Unet(
                model_path=unet_weight_path,
                num_classes=num_classes,
                backbone=backbone,
                input_shape=input_shape,
                mix_type=mix_type,
                cuda=(self.device.type == 'cuda')
            )
            logger.info(f"UNet 加载完成 | 设备: {self.device}")
        except Exception as e:
            raise ValueError(f"UNet模型加载失败: {str(e)}")

    def yolo_detect_crop(self, image_path, crop_save_path=None):
        """
        使用YOLO检测YM目标并裁剪，只返回置信度最高的一个目标

        Args:
            image_path: 输入图片路径
            crop_save_path: 裁剪图保存路径（可选）

        Returns:
            裁剪后的PIL图像，如果没有检测到则返回None
        """
        # 读取原图
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"无法读取图像：{image_path}")
            return None

        # YOLO推理
        results = self.yolo_model.predict(source=image_path)

        best_crop = None
        best_conf = 0

        # 遍历检测到的目标
        for result in results:
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls.cpu().numpy().item())
                class_name = self.yolo_model.names[cls_id]

                if class_name == 'YM':
                    conf = float(box.conf.cpu().numpy().item())

                    # 如果当前检测的置信度更高，更新最佳裁剪
                    if conf > best_conf:
                        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
                        crop = img[y1:y2, x1:x2]

                        if crop.size == 0:
                            continue

                        # 转换为PIL图像
                        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        best_crop = crop_pil
                        best_conf = conf
                        best_bbox = (x1, y1, x2, y2)

        if best_crop is None:
            logger.warning("未检测到YM目标")
            return None

        logger.info(f"已检测到YM目标，置信度：{best_conf:.2f}")

        # 保存裁剪图
        if crop_save_path and best_crop:
            try:
                os.makedirs(os.path.dirname(crop_save_path), exist_ok=True)
                best_crop.save(crop_save_path)
                logger.info(f"裁剪图已保存: {crop_save_path}")
            except Exception as e:
                logger.error(f"裁剪图保存失败: {str(e)}")

        return best_crop

    def analyze(self, image_path, crop_save_path=None, result_save_path=None):
        """
        完整分析流程：YOLO检测 → 裁剪 → UNet分割 → LAB计算

        Args:
            image_path: 输入图片路径
            crop_save_path: 裁剪图保存路径（可选）
            result_save_path: 结果图保存路径（可选）

        Returns:
            {
                'success': True/False,
                'msg': 消息,
                'data': {
                    'L_mean': L均值,
                    'A_mean': A均值,
                    'B_mean': B均值
                },
                'crop_path': 裁剪图路径,
                'result_path': 结果图路径
            }
        """
        try:
            # 步骤1: YOLO检测并裁剪YM目标
            cropped_image = self.yolo_detect_crop(image_path, crop_save_path)

            if cropped_image is None:
                return {
                    'success': False,
                    'msg': '未检测到YM目标',
                    'data': {'L_mean': 0.0, 'A_mean': 0.0, 'B_mean': 0.0},
                    'crop_path': None,
                    'result_path': None
                }

            # 保存原始裁剪图像的副本
            original_image = cropped_image.copy()
            original_array = np.array(original_image)

            # 步骤2: UNet分割
            # 获取分割预测的原始类别索引
            pred_mask = self.unet_model.get_miou_png(original_image)
            pred_mask_array = np.array(pred_mask)

            # 步骤3: 提取果穗区域（GS类别）
            # 创建白色背景
            white_bg = np.ones_like(original_array) * 255

            # 只保留GS类别（果穗）的区域
            gs_mask = (pred_mask_array == GS_CLASS_INDEX)

            # 将原图中的GS区域复制到白色背景上
            white_bg[gs_mask] = original_array[gs_mask]

            # 转换为PIL图像用于保存
            gs_only_image = Image.fromarray(np.uint8(white_bg))

            # 步骤4: 计算LAB均值
            # 创建非白色背景的掩码
            non_white_mask = ~np.all(white_bg == 255, axis=2)

            if not np.any(non_white_mask):
                logger.warning("果穗区域没有非白色像素")
                return {
                    'success': False,
                    'msg': '未识别到果穗区域',
                    'data': {'L_mean': 0.0, 'A_mean': 0.0, 'B_mean': 0.0},
                    'crop_path': crop_save_path,
                    'result_path': None
                }

            # 转换到LAB颜色空间
            gs_only_float = white_bg.astype(np.float32) / 255.0
            gs_only_lab = color.rgb2lab(gs_only_float)

            # 提取非白色像素的LAB值
            l_values = gs_only_lab[:, :, 0][non_white_mask]
            a_values = gs_only_lab[:, :, 1][non_white_mask]
            b_values = gs_only_lab[:, :, 2][non_white_mask]

            # 计算LAB均值
            l_mean = float(np.mean(l_values))
            a_mean = float(np.mean(a_values))
            b_mean = float(np.mean(b_values))

            # 保存结果图像
            result_saved = False
            if result_save_path:
                try:
                    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
                    gs_only_image.save(result_save_path)
                    result_saved = True
                    logger.info(f"结果图像已保存: {result_save_path}")
                except Exception as e:
                    logger.error(f"结果图保存失败: {str(e)}")

            return {
                'success': True,
                'msg': '分析成功',
                'data': {
                    'L_mean': round(l_mean, 2),
                    'A_mean': round(a_mean, 2),
                    'B_mean': round(b_mean, 2)
                },
                'crop_path': crop_save_path,
                'result_path': result_save_path if result_saved else None
            }

        except Exception as e:
            logger.error(f"分析图片时出错: {str(e)}", exc_info=True)
            return {
                'success': False,
                'msg': f'分析失败: {str(e)}',
                'data': {'L_mean': 0.0, 'A_mean': 0.0, 'B_mean': 0.0},
                'crop_path': None,
                'result_path': None
            }


def get_analyzer():
    """获取或创建分析器单例"""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        try:
            _MODEL_CACHE = CornTopColorAnalyzer(
                yolo_weight_path=YOLO_WEIGHT_PATH,
                unet_weight_path=UNET_WEIGHT_PATH,
                num_classes=NUM_CLASSES,
                backbone=BACKBONE,
                input_shape=INPUT_SHAPE,
                mix_type=MIX_TYPE
            )
        except Exception as e:
            logger.error(f"初始化分析器失败: {str(e)}")
            raise
    return _MODEL_CACHE


# --------------------- Flask-RESTX 资源 --------------------- #
@corn_top_color_ns.route('/analyze', methods=['POST'])
class CornTopColorAnalyze(Resource):
    @corn_top_color_ns.doc(
        description='上传玉米图片，分析果穗顶端颜色LAB值',
        responses={
            200: '分析成功',
            400: '无效输入或未识别',
            500: '服务器内部错误'
        }
    )
    def post(self):
        if request.method != 'POST':
            return make_response(jsonify({
                "code": 405,
                "message": "Method Not Allowed"
            }), 405)

        try:
            # 检查文件
            if 'image' not in request.files:
                return make_response(jsonify({
                    "code": 400,
                    "message": "没有图片被上传"
                }), 400)

            image = request.files['image']
            if image.filename == '':
                return make_response(jsonify({
                    "code": 400,
                    "message": "没有选择图片"
                }), 400)

            if not (image and allowed_file(image.filename)):
                return make_response(jsonify({
                    "code": 400,
                    "message": "不支持的文件类型，仅支持: png, jpg, jpeg, bmp"
                }), 400)

            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = secure_filename(image.filename)
            name, ext = os.path.splitext(filename)
            base_name = f"{name}_{timestamp}_{unique_id}"

            # 保存上传文件
            upload_filename = f"{base_name}{ext}"
            upload_path = os.path.join(UPLOAD_FOLDER, upload_filename)
            image.save(upload_path)
            logger.info(f"文件已保存: {upload_path}")

            # 准备输出路径
            crop_path = os.path.join(CROP_FOLDER, f"{base_name}_crop.png")
            result_path = os.path.join(RESULT_FOLDER, f"{base_name}_result.png")

            try:
                # === 核心分析流程 === #
                analyzer = get_analyzer()
                result = analyzer.analyze(
                    upload_path,
                    crop_save_path=crop_path,
                    result_save_path=result_path
                )

                if not result['success']:
                    return make_response(jsonify({
                        "code": 400,
                        "message": result['msg'],
                        "data": result['data']
                    }), 400)

                # 转换为URL路径
                upload_url = convert_to_url_path(upload_path, app_root)

                crop_url = None
                if result.get('crop_path') and os.path.exists(result['crop_path']):
                    crop_url = convert_to_url_path(result['crop_path'], app_root)

                result_url = None
                if result.get('result_path') and os.path.exists(result['result_path']):
                    result_url = convert_to_url_path(result['result_path'], app_root)

                # # 保存历史记录
                # try:
                #     user_id = request.headers.get('token')
                #     if user_id:
                #         history = CornTopColorHistory(
                #             user_id=user_id,
                #             upload_path=upload_url,
                #             crop_path=crop_url,
                #             result_path=result_url,
                #             l_mean=result['data']['L_mean'],
                #             a_mean=result['data']['A_mean'],
                #             b_mean=result['data']['B_mean'],
                #             created_time=datetime.now()
                #         )
                #         db.session.add(history)
                #         db.session.commit()
                #         logger.info(f"历史记录已保存，ID: {history.id}")
                # except Exception as e:
                #     db.session.rollback()
                #     logger.error(f"历史记录保存失败: {str(e)}")

                # 返回成功结果
                return make_response(jsonify({
                    "code": 200,
                    "message": "分析成功",
                    "data": {
                        "L_mean": result['data']['L_mean'],
                        "A_mean": result['data']['A_mean'],
                        "B_mean": result['data']['B_mean'],
                        # "original_image": upload_url,
                        "crop_image": crop_url,
                        "result_image": result_url
                    }
                }), 200)

            except ValueError as ve:
                logger.error(f"分析失败: {str(ve)}")
                return make_response(jsonify({
                    "code": 400,
                    "message": str(ve)
                }), 400)

            except Exception as e:
                logger.error(f"图像分析过程中出错: {str(e)}", exc_info=True)
                return make_response(jsonify({
                    "code": 500,
                    "message": f"图像分析过程中出错: {str(e)}"
                }), 500)

        except Exception as e:
            logger.error(f"服务器内部错误: {str(e)}", exc_info=True)
            return make_response(jsonify({
                "code": 500,
                "message": f"服务器内部错误: {str(e)}",
                "data": None
            }), 500)


# 注册命名空间
api.add_namespace(corn_top_color_ns)