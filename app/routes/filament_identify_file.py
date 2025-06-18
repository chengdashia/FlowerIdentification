import os
import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify, make_response
from flask_restx import Namespace, Resource
from skimage.color import rgb2lab
from app import api
from app.utils.model_loader import load_filament_model
import logging
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

# 创建蓝图和命名空间
filament_identify_file_bp = Blueprint('filament_identify_file', __name__)
filament_identify_file_ns = Namespace('filament_identify_file', description='filament identify api')

# 获取应用根目录的绝对路径
app_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# 配置上传目录
UPLOAD_FOLDER = os.path.join(app_root, 'static', 'images/filament/uploads')
RESULT_FOLDER = os.path.join(app_root, 'static', 'images/filament/results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 加载YOLO模型
unet_model = load_filament_model()


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_comparison_image(original_img, red_region, result_dir, base_name):
    """创建对比图像并保存到文件"""
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

        # 保存图像
        comparison_path = os.path.join(result_dir, f"{base_name}_comparison.jpg")
        comparison_image.save(comparison_path, format='JPEG')

        return comparison_path
    except Exception as e:
        logger.error(f"创建对比图像失败: {e}")
        return None


def process_image(pil_image, result_dir, base_name):
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

        # 6. 保存处理后的图像
        # 保存白色背景图像
        white_bg_path = os.path.join(result_dir, f"{base_name}_white_background.jpg")
        Image.fromarray(white_background).save(white_bg_path, format='JPEG')

        # 保存红色区域图像
        red_region_path = os.path.join(result_dir, f"{base_name}_red_region.jpg")
        Image.fromarray(red_region).save(red_region_path, format='JPEG')

        # 7. 创建对比图像
        comparison_path = create_comparison_image(white_background, red_region, result_dir, base_name)

        return {
            "white_background_path": white_bg_path,
            "red_region_path": red_region_path,
            "comparison_path": comparison_path,
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


@filament_identify_file_ns.route('/analyze', methods=['POST'])
class ImagePredict(Resource):
    @filament_identify_file_ns.doc(
        description='上传图片进行预测和LAB分析',
        responses={
            200: '处理成功',
            400: '无效输入',
            500: '服务器内部错误'
        }
    )
    def post(self):
        try:
            # 检查是否有文件
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

            if image and allowed_file(image.filename):
                # 生成唯一文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                filename = secure_filename(image.filename)
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{timestamp}_{unique_id}{ext}"
                
                # 保存上传的文件
                upload_path = os.path.join(UPLOAD_FOLDER, new_filename)
                image.save(upload_path)
                
                logger.info(f"文件已保存到: {upload_path}")

                # 创建结果目录
                result_dir = os.path.join(RESULT_FOLDER, f"{name}_{timestamp}_{unique_id}")
                os.makedirs(result_dir, exist_ok=True)

                try:
                    # 读取图像
                    pil_image = Image.open(upload_path).convert('RGB')

                    # 处理图像
                    result = process_image(pil_image, result_dir, name)
                    if result is None:
                        return make_response(jsonify({
                            "code": 500,
                            "message": "图像处理失败"
                        }), 500)

                    # 转换为相对路径 - 相对于static目录
                    static_dir = os.path.join(app_root, 'static')
                    white_bg_relative_path = os.path.relpath(result["white_background_path"], static_dir)
                    red_region_relative_path = os.path.relpath(result["red_region_path"], static_dir)
                    comparison_relative_path = os.path.relpath(result["comparison_path"], static_dir) if result["comparison_path"] else None

                    # 返回结果
                    return make_response(jsonify({
                        "code": 200,
                        "message": "处理成功",
                        "data": {
                            "images": {
                                "white_background": f"/static/{white_bg_relative_path.replace(os.sep, '/')}",
                                "red_region": f"/static/{red_region_relative_path.replace(os.sep, '/')}",
                                "comparison": f"/static/{comparison_relative_path.replace(os.sep, '/')}" if comparison_relative_path else None
                            },
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
                    logger.error(f"处理过程中出错: {str(e)}")
                    return make_response(jsonify({
                        "code": 500,
                        "message": f"处理过程中出错: {str(e)}",
                        "data": None
                    }), 500)

            else:
                return make_response(jsonify({
                    "code": 400,
                    "message": "不支持的文件类型"
                }), 400)

        except Exception as e:
            logger.error(f"服务器内部错误: {str(e)}")
            return make_response(jsonify({
                "code": 500,
                "message": f"服务器内部错误: {str(e)}",
                "data": None
            }), 500)


# 注册命名空间
api.add_namespace(filament_identify_file_ns)
