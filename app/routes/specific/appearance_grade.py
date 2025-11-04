import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from app import api
from flask_restx import Namespace, Resource
from flask import request, jsonify, Blueprint, make_response
import logging
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import math

# 导入您的模型
from app.models.corn.specific.grade.unet1 import ImprovedUNet
from app.models.corn.ultralytics import YOLO

logger = logging.getLogger(__name__)

# 创建蓝图和命名空间
corn_grade_bp = Blueprint('corn_grade', __name__)
corn_grade_ns = Namespace('corn_grade', description='玉米高度等级分析API')
# 获取应用根目录的绝对路径
app_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
# 配置上传目录
UPLOAD_FOLDER = os.path.join(app_root, 'static', 'images/corn_grade/uploads')
CROP_FOLDER = os.path.join(app_root, 'static', 'images/corn_grade/crops')
MASK_FOLDER = os.path.join(app_root, 'static', 'images/corn_grade/masks')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 模型路径配置 - 根据实际情况调整
# YOLO_MODEL_PATH = r'static/models/specific/grade/best.pt'
# UNET_WEIGHTS_PATH = r'static/models/specific/grade/best_improved_unet_model.pth'

YOLO_MODEL_PATH = os.path.join(app_root, 'static/models/specific/grade/best.pt')
UNET_WEIGHTS_PATH = os.path.join(app_root, 'static/models/specific/grade/best_improved_unet_model.pth')

# 模型参数
CONF_THRES = 0.25
IOU_THRES = 0.5
IMG_SIZE = 1024
RETINA_MASKS = True
INSTANCE_CONF_MIN = 0.85
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 确保目录存在
for folder in [UPLOAD_FOLDER, CROP_FOLDER, MASK_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 全局分析器实例
_ANALYZER = None


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------- 工具函数 --------------------- #
def get_bbox_from_mask(mask_bin):
    """从二值掩码获取边界框"""
    ys, xs = np.where(mask_bin > 0)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def grade_by_angles(midline_widths, L1, L2, L3):
    """
    根据角度计算等级
    返回: (等级字符串, 角度信息字典)
    """
    if len(midline_widths) != 9:
        return "未识别", {}

    for Li in (L1, L2, L3):
        if Li is None or Li <= 0:
            return "未识别", {}

    W2, W3, W4, W8 = midline_widths[1], midline_widths[2], midline_widths[3], midline_widths[7]

    try:
        cond1 = math.degrees(math.atan((abs(W8 - W2) / 2.0) / float(L1)))
        cond2 = math.degrees(math.atan((abs(W8 - W3) / 2.0) / float(L2)))
        cond3 = math.degrees(math.atan((abs(W8 - W4) / 2.0) / float(L3)))

        # 判断等级
        is_level1 = (cond1 >= 4.0 and cond2 >= 3.0) or \
                    (cond1 >= 4.0 and 2.8 <= cond2 < 3.0 and cond3 >= 2.2)
        is_level3 = (cond1 < 2.0 and cond2 < 2.0 and cond3 < 2.0)

        angles_info = {
            "cond1": round(cond1, 2),
            "cond2": round(cond2, 2),
            "cond3": round(cond3, 2)
        }

        if is_level1:
            return "1级", angles_info
        if is_level3:
            return "3级", angles_info
        return "2级", angles_info

    except Exception as e:
        logger.error(f"计算角度时出错: {str(e)}")
        return "未识别", {}


def analyze_mask_for_grade(mask_path):
    """
    分析掩码图像，计算等级
    返回: (等级, 角度信息, 详细数据)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return "未识别", {}, {}

    height, width = mask.shape
    threshold = 127
    mask_pixels = np.where(mask > threshold)

    if len(mask_pixels[0]) == 0:
        return "未识别", {}, {}

    # 获取掩码的有效区域
    min_row = np.min(mask_pixels[0])
    max_row = np.max(mask_pixels[0])

    # 分成9段
    segment_count = 9
    mask_height = max_row - min_row + 1
    segment_height = mask_height / segment_count

    # 计算每段中线位置和宽度
    midline_positions = []
    midline_widths = []

    for seg in range(segment_count):
        seg_start = min_row + int(seg * segment_height)
        seg_end = min_row + int((seg + 1) * segment_height)
        mid_r = (seg_start + seg_end) // 2
        midline_positions.append(mid_r)
        midline_widths.append(int(np.sum(mask[mid_r, :] > threshold)))

    # 计算关键距离
    if len(midline_positions) >= 8:
        L1 = abs(midline_positions[7] - midline_positions[1])
        L2 = abs(midline_positions[7] - midline_positions[2])
        L3 = abs(midline_positions[7] - midline_positions[3])
    else:
        L1 = L2 = L3 = None

    # 计算等级
    level, angles = grade_by_angles(midline_widths, L1, L2, L3)

    # 详细数据
    detail_data = {
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "midline_widths": midline_widths,
        "mask_height": mask_height
    }

    return level, angles, detail_data


# --------------------- 分析器类 --------------------- #
class IntegratedYMAnalyzer:
    """玉米高度等级分析器"""

    def __init__(self, yolo_model_path, unet_weights_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 加载YOLO模型
        self.yolo_model = YOLO(yolo_model_path)

        # 查找玉米类别ID
        self.ym_class_id = None
        for class_id, class_name in self.yolo_model.names.items():
            if str(class_name).strip().lower() in ("ym", "zl", "corn", "yumi"):
                self.ym_class_id = int(class_id)
                break

        if self.ym_class_id is None:
            logger.warning("未找到玉米类别，将使用class_id=-1")
            self.ym_class_id = -1

        # 加载UNet模型
        self.unet_model = ImprovedUNet(n_channels=3, n_classes=1, n_filters=64)
        self.unet_model.load_state_dict(
            torch.load(unet_weights_path, map_location=self.device)
        )
        self.unet_model.to(self.device).eval()

        # 图像转换
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logger.info(f"分析器初始化完成，使用设备: {self.device}")

    def detect_and_crop_ym(self, image_path, crop_save_path):
        """
        YOLO检测玉米并裁剪第一个高置信度实例
        返回: 裁剪图像路径 或 None
        """
        orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if orig_img is None:
            raise ValueError("无法读取图像")

        if self.ym_class_id < 0:
            raise ValueError("未配置有效的玉米类别ID")

        h, w = orig_img.shape[:2]

        # YOLO检测
        results = self.yolo_model.predict(
            source=orig_img,
            classes=[self.ym_class_id],
            conf=CONF_THRES,
            iou=IOU_THRES,
            imgsz=IMG_SIZE,
            retina_masks=RETINA_MASKS,
            verbose=False
        )

        if not results or results[0].masks is None:
            return None

        result = results[0]
        masks = result.masks.data.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(masks))

        # 筛选高置信度实例
        high_conf_indices = [i for i, conf in enumerate(confs) if conf >= INSTANCE_CONF_MIN]

        if not high_conf_indices:
            return None

        # 只取第一个高置信度实例
        i = high_conf_indices[0]
        mask = masks[i]

        # 调整mask大小
        mh, mw = masks.shape[-2], masks.shape[-1]
        if (mh, mw) != (h, w):
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask

        mask_bin = (mask_resized > 0.5).astype(np.uint8)

        if not mask_bin.any():
            return None

        # 获取边界框并裁剪
        bbox = get_bbox_from_mask(mask_bin)
        if bbox is None:
            return None

        # 创建裁剪图像
        instance_img = orig_img.copy()
        instance_img[mask_bin == 0] = 0

        ys, xs = np.where(mask_bin > 0)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        cropped_img = instance_img[y1:y2 + 1, x1:x2 + 1]

        # 保存裁剪图像
        cv2.imwrite(crop_save_path, cropped_img)
        logger.info(f"裁剪图像已保存: {crop_save_path}")

        return crop_save_path

    def predict_mask(self, image_path, mask_save_path):
        """
        使用UNet预测掩码
        返回: 掩码图像路径
        """
        pil_image = Image.open(image_path).convert('RGB')
        original_size = pil_image.size

        # 转换并预测
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.unet_model(image_tensor)
            if prediction.dim() == 4:
                prediction = prediction.squeeze(1)
            prediction = torch.sigmoid(prediction).squeeze().cpu().numpy()

        # 后处理
        prediction_smooth = cv2.GaussianBlur(prediction, (5, 5), 0)
        binary_mask = (prediction_smooth > 0.5).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.medianBlur(binary_mask, 5)

        # 调整回原始大小
        mask_pil = Image.fromarray(binary_mask).resize(original_size, Image.NEAREST)
        mask_final = np.array(mask_pil)

        # 保存掩码
        cv2.imwrite(mask_save_path, mask_final)
        logger.info(f"掩码图像已保存: {mask_save_path}")

        return mask_save_path

    def analyze_image(self, image_path, crop_save_path, mask_save_path):
        """
        完整分析流程
        返回: {
            'level': '1级/2级/3级/未识别',
            'crop_path': 裁剪图路径,
            'mask_path': 掩码图路径,
            'angles': 角度信息,
            'detail': 详细数据
        }
        """
        # 步骤1: YOLO检测并裁剪
        crop_path = self.detect_and_crop_ym(image_path, crop_save_path)

        if crop_path is None:
            return {
                'level': '未识别',
                'crop_path': None,
                'mask_path': None,
                'angles': {},
                'detail': {},
                'message': '未检测到玉米对象'
            }

        # 步骤2: UNet生成掩码
        mask_path = self.predict_mask(crop_path, mask_save_path)

        # 步骤3: 分析掩码得出等级
        level, angles, detail = analyze_mask_for_grade(mask_path)

        return {
            'level': level,
            'crop_path': crop_path,
            'mask_path': mask_path,
            'angles': angles,
            'detail': detail,
            'message': '分析成功'
        }


def get_analyzer():
    """获取或创建分析器单例"""
    global _ANALYZER
    if _ANALYZER is None:
        try:
            _ANALYZER = IntegratedYMAnalyzer(
                YOLO_MODEL_PATH,
                UNET_WEIGHTS_PATH,
                device=DEVICE
            )
        except Exception as e:
            logger.error(f"初始化分析器失败: {str(e)}")
            raise
    return _ANALYZER


# --------------------- Flask-RESTX 资源 --------------------- #
@corn_grade_ns.route('/analyze', methods=['POST'])
class CornHeightAnalyze(Resource):
    @corn_grade_ns.doc(
        description='上传玉米图片，进行高度等级分析（1级/2级/3级）',
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
            mask_path = os.path.join(MASK_FOLDER, f"{base_name}_mask.png")

            try:
                # === 核心分析流程 === #
                analyzer = get_analyzer()
                result = analyzer.analyze_image(upload_path, crop_path, mask_path)

                level = result['level']

                # 如果未识别到玉米
                if level == '未识别':
                    return make_response(jsonify({
                        "code": 400,
                        "message": result.get('message', '未识别到玉米对象'),
                        "data": {
                            "level": "未识别"
                        }
                    }), 400)

                # 转换为URL路径
                # static_dir = os.path.join(app_root, 'static')
                # upload_url = f"/static/{os.path.relpath(upload_path, static_dir).replace(os.sep, '/')}"
                #
                # crop_url = None
                # mask_url = None
                # if result.get('crop_path'):
                #     crop_url = f"/static/{os.path.relpath(result['crop_path'], static_dir).replace(os.sep, '/')}"
                # if result.get('mask_path'):
                #     mask_url = f"/static/{os.path.relpath(result['mask_path'], static_dir).replace(os.sep, '/')}"

                # # 保存历史记录
                # try:
                #     user_id = request.headers.get('token')
                #     if user_id:
                #         history = CornHeightHistory(
                #             user_id=user_id,
                #             upload_path=upload_url,
                #             crop_path=crop_url,
                #             mask_path=mask_url,
                #             level=level,
                #             angles_cond1=result['angles'].get('cond1'),
                #             angles_cond2=result['angles'].get('cond2'),
                #             angles_cond3=result['angles'].get('cond3'),
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
                        "level": level,
                        "angles": result['angles']
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
api.add_namespace(corn_grade_ns)