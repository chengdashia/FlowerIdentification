import os
import numpy as np
from PIL import Image
import torch
from skimage.color import rgb2lab

from app import api, db
from flask_restx import Namespace, Resource
from flask import request, jsonify, Blueprint, make_response
import logging
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

# 导入您的UNet模型
from app.models.corn.specific.rachis.unet import Unet

logger = logging.getLogger(__name__)

# 创建蓝图和命名空间
corn_rachis_color_bp = Blueprint('corn_rachis_color', __name__)
corn_rachis_color_ns = Namespace('corn_rachis_color', description='玉米穗轴颜色分析API')

# 获取应用根目录的绝对路径
app_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 配置上传目录
UPLOAD_FOLDER = 'static/images/corn_rachis_color/uploads'
RESULT_FOLDER = 'static/images/corn_rachis_color/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# 模型路径配置
UNET_WEIGHT_PATH = r'static/models/specific/rachis/best_epoch_weights.pth'

# 模型参数
NUM_CLASSES = 2  # ["_background_", "0"]
BACKBONE = "vgg"
INPUT_SHAPE = (512, 512)
MIX_TYPE = 2

# 类别列表
CLASSES = ["_background_", "0"]

# 确保目录存在
for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 全局模型缓存
_MODEL_CACHE = None


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------- 核心分析类 --------------------- #
class CornAxisColorAnalyzer:
    """玉米穗轴颜色分析器"""

    def __init__(self, unet_weight_path, num_classes=2, backbone="vgg",
                 input_shape=(512, 512), mix_type=2):
        """
        初始化分析器

        Args:
            unet_weight_path: UNet模型权重路径
            num_classes: 类别数量
            backbone: 骨干网络
            input_shape: 输入图像大小
            mix_type: 混合类型
        """
        if not os.path.exists(unet_weight_path):
            raise FileNotFoundError(f"UNet 权重不存在: {unet_weight_path}")

        self.num_classes = num_classes
        self.cuda = torch.cuda.is_available()

        # 加载UNet模型
        try:
            self.unet_model = Unet(
                model_path=unet_weight_path,
                num_classes=num_classes,
                backbone=backbone,
                input_shape=list(input_shape),
                mix_type=mix_type,
                cuda=self.cuda
            )

            device = "cuda:0" if self.cuda else "cpu"
            logger.info(f"UNet 加载完成 | 设备: {device} | 权重: {unet_weight_path}")

        except Exception as e:
            raise ValueError(f"UNet模型加载失败: {str(e)}")

    def analyze(self, image_path, mask_save_path=None):
        """
        分析单张图片的穗轴颜色

        Args:
            image_path: 输入图片路径
            mask_save_path: 掩码保存路径（可选）

        Returns:
            {
                'success': True/False,
                'msg': 消息,
                'data': {
                    'L_mean': L均值,
                    'A_mean': A均值,
                    'B_mean': B均值
                },
                'mask_path': 掩码图像路径（如果保存）
            }
        """
        try:
            # 检查图片是否存在
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'msg': '图片不存在',
                    'data': {'L_mean': 0.0, 'A_mean': 0.0, 'B_mean': 0.0},
                    'mask_path': None
                }

            # 读取原图
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)

            # UNet分割得到标签图
            label_img = self.unet_model.get_miou_png(image)
            label_map = np.array(label_img, dtype=np.uint8)

            # 提取类别 "0"（索引为1）的掩码
            idx0 = CLASSES.index("0")  # 索引 = 1
            mask0 = (label_map == idx0)

            # 检查是否有前景像素
            if not np.any(mask0):
                return {
                    'success': False,
                    'msg': '未识别到穗轴颖片区域',
                    'data': {'L_mean': 0.0, 'A_mean': 0.0, 'B_mean': 0.0},
                    'mask_path': None
                }

            # 在白底上贴前景
            white_bg = np.ones_like(image_np, dtype=np.uint8) * 255
            white_bg[mask0] = image_np[mask0]

            # 计算前景像素的LAB均值
            # skimage.rgb2lab 期望输入为 [0,1] 浮点
            white_float = white_bg.astype(np.float32) / 255.0
            lab_image = rgb2lab(white_float)

            # 只对前景像素计算均值
            L_mean = float(lab_image[..., 0][mask0].mean())
            a_mean = float(lab_image[..., 1][mask0].mean())
            b_mean = float(lab_image[..., 2][mask0].mean())

            # 保存掩码可视化
            mask_saved = False
            if mask_save_path:
                try:
                    os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
                    Image.fromarray(white_bg).save(mask_save_path)
                    mask_saved = True
                    logger.info(f"掩码图像已保存: {mask_save_path}")
                except Exception as e:
                    logger.error(f"掩码保存失败: {str(e)}")

            return {
                'success': True,
                'msg': '分析成功',
                'data': {
                    'L_mean': round(L_mean, 2),
                    'A_mean': round(a_mean, 2),
                    'B_mean': round(b_mean, 2)
                },
                'mask_path': mask_save_path if mask_saved else None
            }

        except Exception as e:
            logger.error(f"分析图片时出错: {str(e)}", exc_info=True)
            return {
                'success': False,
                'msg': f'分析失败: {str(e)}',
                'data': {'L_mean': 0.0, 'A_mean': 0.0, 'B_mean': 0.0},
                'mask_path': None
            }


def get_analyzer():
    """获取或创建分析器单例"""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        try:
            _MODEL_CACHE = CornAxisColorAnalyzer(
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
@corn_rachis_color_ns.route('/analyze', methods=['POST'])
class CornAxisColorAnalyze(Resource):
    @corn_rachis_color_ns.doc(
        description='上传玉米图片，分析穗轴颜色LAB值',
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

            # 准备掩码输出路径
            mask_path = os.path.join(RESULT_FOLDER, f"{base_name}_mask.png")

            try:
                # === 核心分析流程 === #
                analyzer = get_analyzer()
                result = analyzer.analyze(upload_path, mask_save_path=mask_path)

                if not result['success']:
                    return make_response(jsonify({
                        "code": 400,
                        "message": result['msg'],
                        "data": result['data']
                    }), 400)

                # 转换为URL路径
                static_dir = os.path.join(app_root, 'static')
                # upload_url = f"/static/{os.path.relpath(upload_path, static_dir).replace(os.sep, '/')}"
                mask_url = None
                if result.get('mask_path') and os.path.exists(result['mask_path']):
                    mask_url = f"/static/{os.path.relpath(result['mask_path'], static_dir).replace(os.sep, '/')}"
                #
                # # 保存历史记录
                # try:
                #     user_id = request.headers.get('token')
                #     if user_id:
                #         history = CornAxisColorHistory(
                #             user_id=user_id,
                #             upload_path=upload_url,
                #             mask_path=mask_url,
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
                        "mask_image": mask_url
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
api.add_namespace(corn_rachis_color_ns)