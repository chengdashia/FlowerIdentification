import io
import os
import base64
import torch
import tempfile
from app import db
from datetime import datetime
from flask import Blueprint, request, jsonify, make_response
from flask_restx import Resource, Namespace
from torchvision import transforms
from app import api
from app.models.history import IdentifyHistory
from PIL import Image
from app.utils.model_loader import load_chr_model
from app.utils.image_processing import detect_and_crop
import logging

logger = logging.getLogger(__name__)


# 创建蓝图和命名空间
chr_identify_bp = Blueprint('chr_identify', __name__)
chr_identify_ns = Namespace('chr_identify', description='chrysanthemum identify api')

# 允许的文件类型
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_MIME_TYPES = {'image/png', 'image/jpeg'}

index_to_label1 = {0: "02", 1: "03", 2: "04", 3: "05"}
index_to_label2 = {0: "00", 1: "01", 2: "02"}

# 定义设备
device = torch.device('cpu')

# 加载YOLO模型
yolo_model, net1, net2 = load_chr_model(device)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_mime_type(mime_type):
    return mime_type in ALLOWED_MIME_TYPES


def pil_to_base64(pil_img):
    """将PIL图像转换为base64字符串"""
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str


def predict_with_probabilities(image_path, model, index_to_label):
    """
    对图像进行预测，返回预测标签、最高概率及所有类别的概率
    """
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载图像
    try:
        pil_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"无法打开图像 {image_path}: {e}")
        return None, 0, []

    # 处理图像
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    # 进行预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # 获取所有概率和预测结果
    all_probs = probabilities.squeeze().tolist()
    predicted_index = probabilities.argmax(1).item()
    predicted_label = index_to_label.get(predicted_index, "Unknown")
    probability = probabilities[0][predicted_index].item()

    return predicted_label, probability, all_probs


def predict_image(image_path):
    """对单个图像进行两个模型的预测"""
    global net1, net2, device, index_to_label1, index_to_label2

    try:
        # 检查图像是否存在
        if not os.path.exists(image_path):
            logger.error(f"图像不存在: {image_path}")
            return None

        # 性状1预测
        prediction1, prob1, all_probs1 = predict_with_probabilities(image_path, net1, index_to_label1)
        # 性状2预测
        prediction2, prob2, _ = predict_with_probabilities(image_path, net2, index_to_label2)

        # 根据模型2结果调整模型1预测
        allowed_labels = []
        if prediction2 == "00":
            allowed_labels = ["04", "05"]
        elif prediction2 in ["01", "02"]:
            allowed_labels = ["02", "03"]

        if allowed_labels:
            # 获取允许的索引
            allowed_indices = [idx for idx, label in index_to_label1.items() if label in allowed_labels]

            if allowed_indices:
                # 提取对应概率并找到最大值
                filtered_probs = [all_probs1[idx] for idx in allowed_indices]
                max_prob = max(filtered_probs)
                max_idx = allowed_indices[filtered_probs.index(max_prob)]

                # 更新预测结果
                prediction1 = index_to_label1[max_idx]
                prob1 = max_prob

        # 返回结果
        return {
            "prediction1": prediction1,
            "probability1": float(prob1),  # 转换为float以便JSON序列化
            "prediction2": prediction2,
            "probability2": float(prob2)   # 转换为float以便JSON序列化
        }

    except Exception as e:
        logger.error(f"预测图像时出错: {image_path} - 错误信息: {e}")
        return None


def decode_base64_to_image(base64_data, save_path=None):
    """
    解码base64字符串为图像并可选择保存
    """
    try:
        # 移除头部信息(如 'data:image/jpeg;base64,')
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]

        # 解码base64字符串
        image_data = base64.b64decode(base64_data)

        # 创建PIL图像
        image = Image.open(io.BytesIO(image_data))

        # 如果提供了路径则保存图像
        if save_path:
            image.save(save_path)
            return save_path

        return image

    except Exception as e:
        logger.error(f"解码base64图像时出错: {e}")
        return None


def process_image(file):
    """
    处理上传的图像并返回裁剪后的图像
    """
    try:
        # 读取图像文件并转换为RGB格式
        img_str = file.read()
        image = Image.open(io.BytesIO(img_str)).convert("RGB")

        # 检测并裁剪"front"类别
        front_crops = detect_and_crop(image, yolo_model, target_class=0)
        front_img = front_crops[0] if front_crops else None

        # 检测并裁剪"back"类别
        back_crops = detect_and_crop(image, yolo_model, target_class=1)
        back_img = back_crops[0] if back_crops else None

        # 检查是否找到任何裁剪区域
        if not front_img and not back_img:
            return None, "未检测到目标类别"

        # 准备结果
        results = {
            'front': front_img is not None,
            'back': back_img is not None,
            'msg': "成功裁剪图像"
        }

        # 将图像转换为base64字符串
        if front_img:
            results['front_image'] = pil_to_base64(front_img)

        if back_img:
            results['back_image'] = pil_to_base64(back_img)

        return results, None

    except Exception as e:
        return None, str(e)


@chr_identify_ns.route('/crop', methods=['POST'])
class ImageCrop(Resource):
    @chr_identify_ns.doc(
        description='上传图片并使用YOLO模型裁剪',
        responses={200: '图像处理成功', 400: '无效输入', 500: '服务器内部错误'}
    )
    def post(self):
        """API端点，接收图像文件并返回裁剪结果"""
        if 'file' not in request.files:
            return make_response(jsonify({
                "code": 400,
                "message": "没有文件部分"
            }), 400)

        file = request.files['file']
        if file.filename == '':
            return make_response(jsonify({
                "code": 400,
                "message": "未选择文件"
            }), 400)

        if not allowed_file(file.filename):
            return make_response(jsonify({
                "code": 400,
                "message": "不允许的文件类型"
            }), 400)

        if not allowed_mime_type(file.content_type):
            return make_response(jsonify({
                "code": 400,
                "message": "不允许的MIME类型"
            }), 400)

        # 处理上传的图像
        results, error = process_image(file)

        if error:
            return make_response(jsonify({
                "code": 500,
                "message": "处理图像失败",
                "error": error
            }), 500)

        return make_response(jsonify({
            "code": 200,
            "message": "图像处理成功",
            "result": results
        }), 200)


@chr_identify_ns.route('/predict', methods=['POST'])
class ImagePredict(Resource):
    @chr_identify_ns.doc(
        description='上传BASE64编码的图片并使用RESNET模型预测',
        responses={200: '图像处理成功', 400: '无效输入', 500: '服务器内部错误'}
    )
    def post(self):
        """API端点，接收base64图像并返回预测结果"""
        if request.method == 'POST':
            temp_path = None
            try:
                data = request.get_json()
                headers = request.headers

                # 校验必要字段
                if not data or 'image' not in data:
                    return make_response(jsonify({
                        "code": 400,
                        "message": "缺少图像数据或用户ID"
                    }), 400)

                base64_image = data['image']
                user_id = headers.get('token')
                image_class = data.get('image_class', '')

                # 创建临时文件保存图像
                temp_file_suffix = f"_{image_class}.jpg" if image_class else ".jpg"
                with tempfile.NamedTemporaryFile(suffix=temp_file_suffix, delete=False) as temp_file:
                    temp_path = temp_file.name

                # 解码并保存图像到临时路径
                if not decode_base64_to_image(base64_image, temp_path):
                    return make_response(jsonify({
                        "code": 400,
                        "message": "无法解码图像"
                    }), 400)

                # 调用模型进行预测
                result = predict_image(temp_path)

                # 确保返回包含所有字段
                required_keys = {'prediction1', 'prediction2', 'probability1', 'probability2'}
                if result and required_keys.issubset(result.keys()):
                    # 入库操作
                    history = IdentifyHistory(
                        img=base64_image,
                        user_id=user_id,
                        prediction1=result['prediction1'],
                        probability1=result['probability1'],
                        prediction2=result['prediction2'],
                        probability2=result['probability2'],
                        created_time=datetime.now()
                    )
                    db.session.add(history)
                    db.session.commit()

                    return make_response(jsonify({
                        "code": 200,
                        "message": "识别成功",
                        "result": result
                    }), 200)
                else:
                    return make_response(jsonify({
                        "code": 500,
                        "message": "预测结果格式错误"
                    }), 500)

            except Exception as e:
                db.session.rollback()  # 出错时回滚事务
                return make_response(jsonify({
                    "code": 500,
                    "message": f'处理请求时出错: {str(e)}'
                }), 500)
            finally:
                # 删除临时文件
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.error(f"删除临时文件时出错: {e}")


# 注册命名空间
api.add_namespace(chr_identify_ns)
