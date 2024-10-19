import io
from flask import Blueprint, request, jsonify, make_response
from flask_restx import Resource, Namespace
from app import api
from PIL import Image
import torch
from app.utils.model_loader import load_models
from app.utils.image_processing import detect_and_crop
from app.utils.prediction import predict_with_probabilities

flower_identify_bp = Blueprint('flower_identify', __name__)
flower_identify_ns = Namespace('flower_identify', description='File upload operations')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_MIME_TYPES = {'image/png', 'image/jpeg', 'image/gif'}

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型
net1, net2, yolo_model = load_models(device)

# 标签映射
index_to_label1 = {0: "02", 1: "03", 2: "04", 3: "05"}
index_to_label2 = {0: "00", 1: "01", 2: "02"}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_mime_type(mime_type):
    return mime_type in ALLOWED_MIME_TYPES


def process_image(file):
    try:
        # 读取图片文件并转换为RGB格式
        img_str = file.read()
        image = Image.open(io.BytesIO(img_str)).convert("RGB")

        # YOLO检测裁剪
        front_crops = detect_and_crop(image, yolo_model, target_class=0)
        cropped_img = front_crops[0] if front_crops else None

        if not cropped_img:
            # 尝试检测 'back' 类别
            back_crops = detect_and_crop(image, yolo_model, target_class=1)
            cropped_img = back_crops[0] if back_crops else None

        if not cropped_img:
            return None, "未检测到目标类别"

        # 使用两个BICNN模型进行预测
        traits1 = predict_with_probabilities(cropped_img, net1, device, index_to_label1)
        traits2 = predict_with_probabilities(cropped_img, net2, device, index_to_label2)

        # 返回预测结果
        results = {
            'probabilities': {
                'traits1': traits1['probabilities'],
                'traits2': traits2['probabilities'],
            },
            'predictions': {
                'traits1': traits1['predictions'],
                'traits2': traits2['predictions'],
            },
            'final_classes': traits1['final_classes'] + traits2['final_classes'],
            'msg': f"Detected classes: {', '.join(traits1['final_classes'] + traits2['final_classes'])}"
        }

        return results, None

    except Exception as e:
        return None, str(e)


@flower_identify_ns.route('/image', methods=['POST'])
class UploadImage(Resource):
    @flower_identify_ns.doc(
        description='上传花卉图片并识别',
        responses={200: 'Image processed successfully', 400: 'Invalid input', 500: 'Internal server error'}
    )
    def post(self):
        if 'file' not in request.files:
            return jsonify({"message": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"message": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"message": "File type not allowed"}), 400

        if not allowed_mime_type(file.content_type):
            return jsonify({"message": "MIME type not allowed"}), 400

        # 处理上传的图像并进行分类
        results, error = process_image(file)

        if error:
            return make_response(jsonify({"message": "Failed to process image", "error": error}), 500)

        return make_response(jsonify({
            "code": 200,
            "message": "Image processed successfully",
            "result": results
        }), 200)


# 注册Namespace
api.add_namespace(flower_identify_ns)
