import base64
import io
import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify, make_response
from flask_restx import Namespace, Resource
from skimage.color import rgb2lab
from app import api
from app.utils.model_loader import load_filament_model
import logging

logger = logging.getLogger(__name__)

# 创建蓝图和命名空间
filament_identify_bp = Blueprint('filament_identify', __name__)
filament_identify_ns = Namespace('filament_identify', description='filament identify api')

# 加载YOLO模型
unet_model = load_filament_model()


def decode_base64_to_pil(base64_str):
    """base64 -> PIL Image"""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[-1]
        img_bytes = base64.b64decode(base64_str)
        img_pil = Image.open(io.BytesIO(img_bytes))
        return img_pil
    except Exception as e:
        logger.info(f"[decode_base64_to_pil] 失败: {e}")
        return None


def encode_image_to_base64(img_array, format='JPEG'):
    """numpy array -> base64"""
    try:
        img_pil = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img_pil.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_str
    except Exception as e:
        logger.error(f"[encode_image_to_base64] 失败: {e}")
        return None


def create_comparison_image(original_img, red_region):
    """使用PIL创建对比图像"""
    try:
        # 确保输入是numpy数组并转换为PIL Image
        if isinstance(original_img, np.ndarray):
            original_pil = Image.fromarray(original_img)
        else:
            original_pil = original_img

        if isinstance(red_region, np.ndarray):
            red_region_pil = Image.fromarray(red_region)
        else:
            red_region_pil = red_region

        # 获取单个图像的尺寸
        width, height = original_pil.size

        # 创建新图像（宽度是原图的2倍，为两张图并排）
        comparison_image = Image.new('RGB', (width * 2, height), (255, 255, 255))

        # 粘贴两张图
        comparison_image.paste(original_pil, (0, 0))  # 左边
        comparison_image.paste(red_region_pil, (width, 0))  # 右边

        # 转换为base64
        buffer = io.BytesIO()
        comparison_image.save(buffer, format='PNG')
        comparison_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return comparison_base64
    except Exception as e:
        logger.error(f"创建对比图像失败: {e}")
        return None


def process_image(pil_image):
    """处理图像并返回结果"""
    try:
        # 1. 模型预测和初始处理
        result = unet_model.detect_image(pil_image)
        image_rgb = pil_image.convert('RGB')
        image_np = np.array(image_rgb)
        pred_np = np.array(result)
        mask = np.any(pred_np != [0, 0, 0], axis=-1)
        white_background = np.ones_like(image_np, dtype=np.uint8) * 255
        white_background[mask] = image_np[mask]

        # 2. 计算LAB值和提取红色区域
        image_rgb_norm = white_background / 255.0
        image_lab = rgb2lab(image_rgb_norm)
        non_white_mask = ~np.all(white_background == [255, 255, 255], axis=-1)
        if not np.any(non_white_mask):
            raise ValueError("没有有效的非白色像素")

        # 3. 计算原始图像LAB均值
        original_lab_mean = np.mean(image_lab[non_white_mask], axis=0)

        # 4. 提取红色区域
        a_channel = image_lab[:, :, 1]
        red_mask = a_channel > 10
        red_region = np.zeros_like(white_background)
        red_region[red_mask] = white_background[red_mask]

        # 5. 计算红色区域LAB均值
        if np.any(red_mask):
            red_lab_mean = np.mean(image_lab[red_mask], axis=0)
        else:
            red_lab_mean = np.array([np.nan, np.nan, np.nan])

        # 6. 创建对比图像
        comparison_base64 = create_comparison_image(white_background, red_region)

        return {
            "comparison_image": comparison_base64,
            "original_lab": {
                "L": round(float(original_lab_mean[0]), 2),
                "A": round(float(original_lab_mean[1]), 2),
                "B": round(float(original_lab_mean[2]), 2)
            },
            "red_region_lab": {
                "L": round(float(red_lab_mean[0]), 2) if not np.isnan(red_lab_mean[0]) else None,
                "A": round(float(red_lab_mean[1]), 2) if not np.isnan(red_lab_mean[1]) else None,
                "B": round(float(red_lab_mean[2]), 2) if not np.isnan(red_lab_mean[2]) else None
            }
        }

    except Exception as e:
        logger.error(f"处理图像时出错: {str(e)}")
        return None


@filament_identify_ns.route('/analyze', methods=['POST'])
class ImagePredict(Resource):
    @filament_identify_ns.doc(
        description='上传图片进行预测和LAB分析',
        responses={
            200: '处理成功',
            400: '无效输入',
            500: '服务器内部错误'
        }
    )
    def post(self):
        try:
            # 检查请求
            if not request.is_json:
                return make_response(jsonify({
                    "code": 400,
                    "message": "请求必须是JSON格式"
                }), 400)

            data = request.get_json()
            if 'image' not in data:
                return make_response(jsonify({
                    "code": 400,
                    "message": "缺少图像数据"
                }), 400)

            # 解码图像
            pil_image = decode_base64_to_pil(data['image'])
            if pil_image is None:
                return make_response(jsonify({
                    "code": 400,
                    "message": "无法解码图像"
                }), 400)

            # 处理图像
            result = process_image(pil_image)
            if result is None:
                return make_response(jsonify({
                    "code": 500,
                    "message": "图像处理失败"
                }), 500)

            # 返回结果
            return make_response(jsonify({
                "code": 200,
                "message": "处理成功",
                "data": {
                    "comparison_image": result["comparison_image"],
                    "lab_values": {
                        "original": {
                            "L": result['original_lab']['L'],
                            "A": result['original_lab']['A'],
                            "B": result['original_lab']['B']
                        },
                        "red_region": {
                            "L": result['red_region_lab']['L'] if result['red_region_lab']['L'] is not None else 0,
                            "A": result['red_region_lab']['A'] if result['red_region_lab']['A'] is not None else 0,
                            "B": result['red_region_lab']['B'] if result['red_region_lab']['B'] is not None else 0
                        }
                    }
                }
            }), 200)

        except Exception as e:
            return make_response(jsonify({
                "code": 500,
                "message": f"服务器内部错误: {str(e)}"
            }), 500)


# 注册命名空间
api.add_namespace(filament_identify_ns)
