import os
import cv2
import numpy as np
from app import api, db
from app.models.history import CornHistory
from flask_restx import Namespace, Resource
from flask import request, jsonify, Blueprint, make_response
import logging
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

# 创建蓝图和命名空间
corn_filament_bp = Blueprint('corn_filament', __name__)
corn_filament_ns = Namespace('corn_filament', description='corn filament identify api')

# 获取应用根目录的绝对路径
app_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# 配置上传目录
UPLOAD_FOLDER = os.path.join(app_root, 'static', 'images/corn/uploads')
RESULT_FOLDER = os.path.join(app_root, 'static', 'images/corn/results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --------------------- 图像处理相关函数 --------------------- #
def remove_gray_background(image, threshold=40):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(image_rgb)
    max_rgb = np.maximum(np.maximum(R, G), B)
    min_rgb = np.minimum(np.minimum(R, G), B)
    diff = max_rgb - min_rgb
    mask = diff < threshold
    image_rgb[mask] = [255, 255, 255]
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def remove_green_region(image, lower_green, upper_green):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_DILATE, kernel, iterations=1)
    result = image.copy()
    result[mask_cleaned > 0] = [255, 255, 255]
    return result


def calc_lab_stats(image_bgr):
    """
    使用 OpenCV 计算 LAB，避免 skimage 内部 float64 占用
    返回:
        mean_lab, max_a_lab
    """
    # ---------- ① 转 LAB（uint8） ----------
    lab_u8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    # OpenCV 范围：
    #   L: 0~255 → 真正 L ≈ (L/255)*100
    #   a: 0~255 → 真正 a ≈ a - 128
    #   b: 0~255 → 真正 b ≈ b - 128

    # ---------- ② 过滤白色像素 ----------
    white_mask = np.all(image_bgr == [255, 255, 255], axis=-1)
    valid_mask = ~white_mask
    if not np.any(valid_mask):
        return None, None

    # ---------- ③ 统计 ----------
    L = lab_u8[:, :, 0][valid_mask].astype(np.float32) * 100.0 / 255
    A = lab_u8[:, :, 1][valid_mask].astype(np.float32) - 128.0
    B = lab_u8[:, :, 2][valid_mask].astype(np.float32) - 128.0

    mean_lab = {"L": round(float(np.mean(L)), 2),
                "A": round(float(np.mean(A)), 2),
                "B": round(float(np.mean(B)), 2)}

    max_idx = np.argmax(A)
    max_a_lab = {"L": round(float(L[max_idx]), 2),
                 "A": round(float(A[max_idx]), 2),
                 "B": round(float(B[max_idx]), 2)}
    return mean_lab, max_a_lab


# --------------------- 业务主流程 --------------------- #
def pipeline_process(image_path, result_dir):
    """
    返回:
        processed_image_path, mean_lab(dict), max_a_lab(dict)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError("无法读取图像")

    # 1. 去灰背景
    img_no_gray = remove_gray_background(img_bgr, threshold=40)

    # 2. 去绿色
    lower_green = np.array([22, 40, 40])
    upper_green = np.array([85, 255, 255])
    img_clean = remove_green_region(img_no_gray, lower_green, upper_green)

    # 3. 计算 LAB
    mean_lab, max_a_lab = calc_lab_stats(img_clean)
    if mean_lab is None:
        raise ValueError("所选区域无有效像素（非白色）")

    # 4. 保存处理后的图像
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    processed_image_path = os.path.join(result_dir, f"{base_name}_processed.jpg")
    cv2.imwrite(processed_image_path, img_clean)

    return processed_image_path, mean_lab, max_a_lab


# --------------------- Flask-RESTX 资源 --------------------- #
@corn_filament_ns.route('/analyze', methods=['POST'])
class ColorAnalyze(Resource):
    @corn_filament_ns.doc(
        description='上传图片文件，去灰/去绿并计算 LAB',
        responses={
            200: '图像处理成功',
            400: '无效输入',
            500: '服务器内部错误'
        }
    )
    def post(self):
        if request.method != 'POST':
            return make_response(jsonify({"code": 405, "message": "Method Not Allowed"}), 405)

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
                    # === 业务主流程 === #
                    processed_image_path, mean_lab, max_a_lab = pipeline_process(upload_path, result_dir)

                    # 转换为相对路径 - 相对于static目录
                    static_dir = os.path.join(app_root, 'static')
                    processed_relative_path = os.path.relpath(processed_image_path, static_dir)
                    processed_url = f"/static/{processed_relative_path.replace(os.sep, '/')}"
                    upload_url = f"/static/{os.path.relpath(upload_path, static_dir).replace(os.sep, '/')}"

                    # 入库操作
                    try:
                        user_id = request.headers.get('token')
                        if user_id:
                            history = CornHistory(
                                user_id=user_id,
                                upload_path=upload_url,
                                processed_image_path=processed_url,
                                mean_lab_l=mean_lab['L'],
                                mean_lab_a=mean_lab['A'],
                                mean_lab_b=mean_lab['B'],
                                max_a_lab_l=max_a_lab['L'],
                                max_a_lab_a=max_a_lab['A'],
                                max_a_lab_b=max_a_lab['B'],
                                created_time=datetime.now()
                            )
                            db.session.add(history)
                            db.session.commit()
                    except Exception as e:
                        db.session.rollback()
                        logger.error(f"历史记录保存失败: {str(e)}")


                    return make_response(jsonify({
                        "code": 200,
                        "message": "图像处理成功",
                        "data": {
                            "processed_image": processed_url,
                            "mean_lab": mean_lab,
                            "max_a_lab": max_a_lab
                        }
                    }), 200)

                except ValueError as ve:
                    # 业务可预见错误
                    return make_response(jsonify({
                        "code": 400,
                        "message": str(ve)
                    }), 400)
                except Exception as e:
                    # 未知错误
                    return make_response(jsonify({
                        "code": 500,
                        "message": f"图像处理过程中出错: {str(e)}"
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
api.add_namespace(corn_filament_ns)
