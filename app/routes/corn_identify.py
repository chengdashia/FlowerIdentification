import base64
import cv2
import numpy as np
from app import api
from flask_restx import Namespace, Resource
from flask import request, jsonify, Blueprint, make_response
from skimage.color import rgb2lab


# 创建蓝图和命名空间
corn_identify_bp = Blueprint('corn_identify', __name__)
corn_identify_ns = Namespace('corn_identify', description='corn identify api')


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
    返回：
        mean_lab  : dict{'L','A','B'}
        max_a_lab : dict{'L','A','B'}
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_lab = rgb2lab(image_rgb / 255.0)

    # 去掉白色像素
    white_mask = np.all(image_rgb == [255, 255, 255], axis=-1)
    valid_mask = ~white_mask

    if not np.any(valid_mask):
        # 整张图都是白色，返回空结果
        return None, None

    valid_L = image_lab[:, :, 0][valid_mask]
    valid_A = image_lab[:, :, 1][valid_mask]
    valid_B = image_lab[:, :, 2][valid_mask]

    mean_L = float(np.mean(valid_L))
    mean_A = float(np.mean(valid_A))
    mean_B = float(np.mean(valid_B))

    max_idx = np.argmax(valid_A)
    max_L = float(valid_L[max_idx])
    max_A = float(valid_A[max_idx])
    max_B = float(valid_B[max_idx])

    mean_lab = {"L": round(mean_L, 2), "A": round(mean_A, 2), "B": round(mean_B, 2)}
    max_a_lab = {"L": round(max_L, 2), "A": round(max_A, 2), "B": round(max_B, 2)}
    return mean_lab, max_a_lab


# --------------------- 编解码工具函数 --------------------- #
def decode_base64_to_image(base64_str):
    """base64 -> OpenCV BGR 图像，失败返回 None"""
    try:
        # 兼容 data URI 头
        if "," in base64_str:
            base64_str = base64_str.split(",")[-1]
        img_bytes = base64.b64decode(base64_str)
        img_np = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        return img_bgr
    except Exception as e:
        print(f"[decode_base64_to_image] 失败: {e}")
        return None


def encode_image_to_base64(img_bgr, ext=".jpg"):
    """OpenCV BGR -> base64 字符串（不带 data URI 头）"""
    success, buffer = cv2.imencode(ext, img_bgr)
    if not success:
        return None
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return base64_str


# --------------------- 业务主流程 --------------------- #
def pipeline_process(base64_img):
    """
    返回:
        processed_b64, mean_lab(dict), max_a_lab(dict)
    """
    img_bgr = decode_base64_to_image(base64_img)
    if img_bgr is None:
        raise ValueError("无法解码图像")

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

    # 4. 编码回 base64
    processed_b64 = encode_image_to_base64(img_clean)
    if processed_b64 is None:
        raise ValueError("图像编码失败")

    return processed_b64, mean_lab, max_a_lab


# --------------------- Flask-RESTX 资源 --------------------- #
@corn_identify_ns.route('/analyze', methods=['POST'])
class ColorAnalyze(Resource):
    @corn_identify_ns.doc(
        description='上传 BASE64 图片，去灰/去绿并计算 LAB',
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
            data = request.get_json(silent=True)
            if not data or 'image' not in data:
                return make_response(jsonify({
                    "code": 400,
                    "message": "缺少图像数据"
                }), 400)

            base64_image = data['image']

            # === 业务主流程 === #
            processed_b64, mean_lab, max_a_lab = pipeline_process(base64_image)

            # （可选）这里可以像原接口一样写入数据库
            # save_history(...)

            return make_response(jsonify({
                "code": 200,
                "message": "图像处理成功",
                "data": {
                    "processed_image": processed_b64,
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
                "message": f"服务器内部错误: {str(e)}"
            }), 500)


# 注册命名空间
api.add_namespace(corn_identify_ns)
