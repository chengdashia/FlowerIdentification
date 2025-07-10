import os
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
from flask import Blueprint, request, jsonify, make_response
from flask_restx import Namespace, Resource
from app import api, db
from app.models.history import YmHistory
import logging
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
from app.utils.model_loader import load_ym_models
from skimage import color

logger = logging.getLogger(__name__)

# 创建蓝图和命名空间
ym_all_analyzer_bp = Blueprint('ym_all_analyzer', __name__)
ym_all_analyzer_ns = Namespace('ym_all_analyzer', description='YM All Shape Analysis API  直接上传图片进行分析')

# 获取应用根目录的绝对路径
app_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# 配置上传目录
UPLOAD_FOLDER = os.path.join(app_root, 'static', 'images/ym/uploads')
RESULT_FOLDER = os.path.join(app_root, 'static', 'images/ym/results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 加载模型
yolo_model, unet_model, zl_model, ym_unet_model = load_ym_models()

# 定义预处理转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

sk_short_cm = 5.5
sk_long_cm = 7.85
class_names = ['0', 'ZL', 'SK', 'YM']


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_mask_shape(midline_widths):
    if len(midline_widths) != 5:
        return "数据不足"

    # 避免除零错误
    if midline_widths[2] == 0 or midline_widths[4] == 0:
        return "无法分析（除零错误）"

    # 计算比值
    ratio_1_3 = midline_widths[0] / midline_widths[2]  # 1号与3号比值
    ratio_3_5 = midline_widths[2] / midline_widths[4]  # 3号与5号比值
    ratio_1_5 = midline_widths[0] / midline_widths[4]  # 1号与5号比值

    # 根据条件判断形状
    if ratio_1_3 < 0.7 and ratio_3_5 < 0.8 and ratio_1_5 < 0.6:
        return "锥形"
    elif ratio_1_3 > 0.8 and ratio_3_5 > 0.8 and ratio_1_5 > 0.8:
        return "筒形"
    elif (0.7 <= ratio_1_3 <= 0.8) and ratio_3_5 > 0.8 and (0.6 <= ratio_1_5 <= 0.8):
        return "锥到筒形"
    else:
        return "其他形状"


def split_mask_horizontally(mask_array, add_midlines=True, midline_thickness=2, analyze_shape=True):
    """
    将掩码图的掩码部分进行横向五等分，并进行形状分析
    """
    # 确保掩码是灰度图
    if len(mask_array.shape) > 2:
        mask = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
    else:
        mask = mask_array

    height, width = mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    colors = [
        (255, 200, 200),  # 浅红色
        (200, 255, 200),  # 浅绿色
        (200, 200, 255),  # 浅蓝色
        (255, 255, 200),  # 浅黄色
        (255, 200, 255),  # 浅紫色
    ]

    # 中位线颜色 (紫色)
    midline_color = (128, 0, 128)

    # 找到掩码区域的边界
    threshold = 127
    mask_pixels = np.where(mask > threshold)

    if len(mask_pixels[0]) == 0:
        logger.error("警告：未找到掩码区域")
        return colored_mask, "无掩码区域", []

    # 获取掩码区域的边界
    min_row = np.min(mask_pixels[0])
    max_row = np.max(mask_pixels[0])

    # 计算每个分段的高度
    mask_height = max_row - min_row + 1
    segment_height = mask_height / 5

    # 对每个像素进行分类和着色
    for i in range(len(mask_pixels[0])):
        row = mask_pixels[0][i]
        col = mask_pixels[1][i]

        relative_row = row - min_row
        segment_index = int(relative_row / segment_height)
        segment_index = min(segment_index, 4)

        original_alpha = mask[row, col] / 255.0
        base_color = np.array(colors[segment_index])
        final_color = base_color * original_alpha

        colored_mask[row, col] = final_color.astype(np.uint8)

    # 添加中位线和数字编号
    midline_widths = []

    if add_midlines:
        midline_positions = []

        for segment in range(5):
            segment_start_row = min_row + int(segment * segment_height)
            segment_end_row = min_row + int((segment + 1) * segment_height)
            midline_row = (segment_start_row + segment_end_row) // 2

            midline_positions.append(midline_row)

            mask_pixels_in_row = np.sum(mask[midline_row, :] > threshold)
            midline_widths.append(mask_pixels_in_row)

            # 绘制中位线
            for col in range(width):
                if mask[midline_row, col] > threshold:
                    for thickness_offset in range(-midline_thickness // 2, midline_thickness // 2 + 1):
                        line_row = midline_row + thickness_offset
                        if 0 <= line_row < height and mask[line_row, col] > threshold:
                            colored_mask[line_row, col] = midline_color

        # 添加数字编号
        pil_image = Image.fromarray(cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # 使用默认字体
        font = ImageDraw.Draw(pil_image).font

        for i, midline_row in enumerate(midline_positions):
            mask_cols_in_row = np.where(mask[midline_row, :] > threshold)[0]
            if len(mask_cols_in_row) > 0:
                center_col = (mask_cols_in_row[0] + mask_cols_in_row[-1]) // 2
                number_text = str(i + 1)

                # 估计文本大小
                text_width = 20
                text_height = 20

                text_x = center_col - text_width // 2
                text_y = midline_row - text_height // 2

                circle_radius = max(text_width, text_height) // 2 + 5
                draw.ellipse([
                    center_col - circle_radius,
                    midline_row - circle_radius,
                    center_col + circle_radius,
                    midline_row + circle_radius
                ], fill='white', outline='black', width=2)

                draw.text((text_x, text_y), number_text, fill='black')

        colored_mask = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    else:
        # 不显示中位线时仍计算中位线宽度
        for segment in range(5):
            segment_start_row = min_row + int(segment * segment_height)
            segment_end_row = min_row + int((segment + 1) * segment_height)
            midline_row = (segment_start_row + segment_end_row) // 2

            mask_pixels_in_row = np.sum(mask[midline_row, :] > threshold)
            midline_widths.append(mask_pixels_in_row)

    # 形状分析
    shape_result = ""
    if analyze_shape and len(midline_widths) == 5:
        shape_result = analyze_mask_shape(midline_widths)

    return colored_mask, shape_result, midline_widths


class IntegratedYMAnalyzer:
    def __init__(self, device='cpu'):
        self.device = torch.device('cpu')
        self.yolo_model = yolo_model
        self.unet_model = unet_model
        self.zl_model = zl_model
        self.ym_unet_model = ym_unet_model
        self.transform = transform

    def yolo_detect_crop(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"无法读取图像：{image_path}")
            return None, 0.0
        results = self.yolo_model.predict(source=image_path, device='cpu')

        best_crop = None
        best_conf = 0.0

        for result in results:
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls.cpu().numpy().item())
                class_name = self.yolo_model.names[cls_id]
                if class_name == 'YM':
                    conf = float(box.conf.cpu().numpy().item())

                    if conf > best_conf:
                        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
                        crop = img[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        best_crop = crop_pil
                        best_conf = conf

        if best_crop is None:
            logger.error(f"未检测到YM目标")
        else:
            logger.info(f"已检测到YM目标，置信度：{best_conf:.2f}")

        return best_crop, best_conf

    def process_single_image(self, image_path, result_dir):
        import matplotlib.pyplot as plt
        try:
            cropped_image, best_conf = self.yolo_detect_crop(image_path)

            if cropped_image is None:
                logger.error("未检测到YM目标，无法进行后续处理")
                return None

            original_full = cv2.imread(image_path)
            original_full = cv2.cvtColor(original_full, cv2.COLOR_BGR2RGB)

            original_image = cropped_image.copy()
            original_array = np.array(original_image)

            r_image = self.ym_unet_model.detect_image(cropped_image)

            GS_CLASS_INDEX = 1

            pred_mask = self.ym_unet_model.get_miou_png(original_image)
            pred_mask_array = np.array(pred_mask)

            white_bg = np.ones_like(original_array) * 255

            gs_mask = (pred_mask_array == GS_CLASS_INDEX)

            white_bg[gs_mask] = original_array[gs_mask]

            gs_only_image = Image.fromarray(np.uint8(white_bg))

            binary_mask = np.zeros_like(gs_mask, dtype=np.uint8)
            binary_mask[gs_mask] = 255

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 保存图片路径
            os.makedirs(result_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            gs_only_image_path = os.path.join(result_dir, f"{base_name}_gs_only_image.jpg")
            r_image_path = os.path.join(result_dir, f"{base_name}_unet_result.jpg")
            center_part_path = os.path.join(result_dir, f"{base_name}_center_part.jpg")

            gs_only_image.save(gs_only_image_path)
            r_image.save(r_image_path)

            if contours:
                max_contour = max(contours, key=cv2.contourArea)

                x, y, w, h = cv2.boundingRect(max_contour)

                x1 = x + w // 4
                x2 = x + 3 * w // 4
                y1 = y + h // 4
                y2 = y + 3 * h // 4

                highlight_image = np.array(gs_only_image.copy())

                cv2.rectangle(highlight_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

                cv2.line(highlight_image, (x, y1), (x + w, y1), (0, 0, 255), 2)
                cv2.line(highlight_image, (x, y2), (x + w, y2), (0, 0, 255), 2)
                cv2.line(highlight_image, (x1, y), (x1, y + h), (0, 0, 255), 2)
                cv2.line(highlight_image, (x2, y), (x2, y + h), (0, 0, 255), 2)

                overlay = highlight_image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)  # 填充蓝色矩形

                alpha = 0.4
                cv2.addWeighted(overlay, alpha, highlight_image, 1 - alpha, 0, highlight_image)

                highlight_image_pil = Image.fromarray(highlight_image)
                highlight_image_path = os.path.join(result_dir, f"{base_name}_highlight.jpg")
                highlight_image_pil.save(highlight_image_path)

                center_part = np.array(gs_only_image)[y1:y2, x1:x2]
                center_part_pil = Image.fromarray(center_part)
                center_part_pil.save(center_part_path)

                # 计算LAB均值和最大值
                non_white_mask = ~np.all(center_part == 255, axis=2)
                lab_mean = None
                lab_max = None
                if np.any(non_white_mask):
                    center_part_float = center_part.astype(np.float32) / 255.0
                    center_part_lab = color.rgb2lab(center_part_float)

                    l_values = center_part_lab[:, :, 0][non_white_mask]
                    a_values = center_part_lab[:, :, 1][non_white_mask]
                    b_values = center_part_lab[:, :, 2][non_white_mask]

                    l_mean = float(np.mean(l_values))
                    a_mean = float(np.mean(a_values))
                    b_mean = float(np.mean(b_values))

                    l_max = float(np.max(l_values))
                    a_max = float(np.max(a_values))
                    b_max = float(np.max(b_values))

                    lab_mean = {'L': l_mean, 'A': a_mean, 'B': b_mean}
                    lab_max = {'L': l_max, 'A': a_max, 'B': b_max}
                return {
                    'gs_only_image': gs_only_image_path,
                    'unet_result': r_image_path,
                    'highlight': highlight_image_path,
                    'center_part': center_part_path,
                    'confidence': best_conf,
                    'lab_mean': lab_mean,
                    'lab_max': lab_max
                }
            else:
                logger.error(f"未检测到果穗轮廓")
                gs_only_image.save(gs_only_image_path)
                r_image.save(r_image_path)
                return {
                    'gs_only_image': gs_only_image_path,
                    'unet_result': r_image_path,
                    'highlight': None,
                    'center_part': None,
                    'confidence': best_conf,
                    'lab_mean': None,
                    'lab_max': None
                }
        except Exception as e:
            logger.error(f"处理图片时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def detect_and_crop_ym(self, image_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        results = self.yolo_model.predict(source=image_path)
        img = cv2.imread(image_path)

        if img is None:
            logger.error(f"❌ 无法读取图像：{image_path}")
            return []

        cropped_paths = []
        ym_crops = []

        for result in results:
            if result.boxes is None:
                continue

            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls.cpu().item())
                class_name = self.yolo_model.names[cls_id]

                if class_name == 'YM':
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
                    crop = img[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    base, _ = os.path.splitext(os.path.basename(image_path))
                    out_name = f"{base}_YM_{i}.png"
                    out_path = os.path.join(output_dir, out_name)

                    cv2.imwrite(out_path, crop)
                    cropped_paths.append(out_path)

                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    ym_crops.append((crop_rgb, out_name))

        return cropped_paths

    def predict_mask(self, image_path):
        """对单张图像进行掩码预测"""
        pil_image = Image.open(image_path).convert('RGB')
        original_size = pil_image.size

        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.unet_model(image_tensor)
            if prediction.dim() == 4:
                prediction = prediction.squeeze(1)
            prediction = torch.sigmoid(prediction)
            prediction = prediction.squeeze().cpu().numpy()

        binary_mask = (prediction > 0.5).astype(np.uint8) * 255

        mask_pil = Image.fromarray(binary_mask)
        mask_pil = mask_pil.resize((original_size[0], original_size[1]), Image.NEAREST)
        mask_array = np.array(mask_pil)

        return mask_array

    def process_cropped_images_with_shape_analysis(self, cropped_paths, result_dir):
        """
        对裁剪后的图像进行掩码检测和形状分析
        """
        os.makedirs(result_dir, exist_ok=True)

        analysis_results = []  # 存储所有分析结果

        for i, crop_path in enumerate(cropped_paths):
            # 预测掩码
            predicted_mask = self.predict_mask(crop_path)

            # 加载原始裁剪图像
            original_image = np.array(Image.open(crop_path).convert('RGB'))

            # 确保掩码尺寸匹配
            if predicted_mask.shape[:2] != original_image.shape[:2]:
                mask_pil = Image.fromarray(predicted_mask)
                mask_pil = mask_pil.resize((original_image.shape[1], original_image.shape[0]), Image.NEAREST)
                predicted_mask = np.array(mask_pil)

            # 创建叠加效果
            color_mask = np.zeros_like(original_image)
            color_mask[predicted_mask > 0] = [255, 165, 0]  # 橙色掩码
            cv_overlay = cv2.addWeighted(original_image, 1, color_mask, 0.7, 0)

            # 保存基础结果
            base_name = os.path.splitext(os.path.basename(crop_path))[0]

            # 保存掩码
            mask_path = os.path.join(result_dir, f"{base_name}_mask.jpg")
            cv2.imwrite(mask_path, predicted_mask)

            # 保存叠加结果
            overlay_path = os.path.join(result_dir, f"{base_name}_overlay.jpg")
            cv2.imwrite(overlay_path, cv2.cvtColor(cv_overlay, cv2.COLOR_RGB2BGR))

            # 进行形状分析
            try:
                colored_mask, shape_result, midline_widths = split_mask_horizontally(
                    predicted_mask,
                    add_midlines=True,
                    midline_thickness=2,
                    analyze_shape=True
                )

                # 保存形状分析结果
                analysis_path = os.path.join(result_dir, f"{base_name}_shape_analysis.jpg")
                cv2.imwrite(analysis_path, cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB))

                # 存储分析结果
                analysis_result = {
                    'image_name': base_name,
                    'crop_path': crop_path,
                    'shape_type': shape_result,
                    'midline_widths': midline_widths,
                    'mask_path': mask_path,
                    'overlay_path': overlay_path,
                    'analysis_path': analysis_path
                }
                analysis_results.append(analysis_result)

                # 输出形状分析结果
                logger.info(f"\n=== {base_name} 形状分析结果 ===")
                logger.info(f"各中位线宽度: {midline_widths}")

                if len(midline_widths) == 5 and midline_widths[2] != 0 and midline_widths[4] != 0:
                    ratio_1_3 = midline_widths[0] / midline_widths[2]
                    ratio_3_5 = midline_widths[2] / midline_widths[4]
                    ratio_1_5 = midline_widths[0] / midline_widths[4]

                    logger.info(f"1号与3号比值: {ratio_1_3:.3f}")
                    logger.info(f"3号与5号比值: {ratio_3_5:.3f}")
                    logger.info(f"1号与5号比值: {ratio_1_5:.3f}")

                logger.info(f"形状判断结果: {shape_result}")
                logger.info("=" * 30)

            except Exception as e:
                logger.error(f"❌ 形状分析失败: {e}")
                analysis_result = {
                    'image_name': base_name,
                    'crop_path': crop_path,
                    'shape_type': f"分析失败: {e}",
                    'midline_widths': [],
                    'mask_path': mask_path,
                    'overlay_path': overlay_path,
                    'analysis_path': None
                }
                analysis_results.append(analysis_result)

        return analysis_results

    def run_complete_analysis_pipeline(self, image_path, result_dir="analysis_results"):
        """
        运行完整的分析流程：检测 -> 裁剪 -> 掩码预测 -> 形状分析
        """
        # 步骤1: 检测并裁剪YM区域
        cropped_paths = self.detect_and_crop_ym(image_path, result_dir)

        if not cropped_paths:
            logger.error("❌ 未检测到YM区域，分析结束")
            return []

        # 步骤2: 掩码预测与形状分析
        analysis_results = self.process_cropped_images_with_shape_analysis(cropped_paths, result_dir)

        return analysis_results

    def calc_zl_length_single(self, input_image):
        image_name = os.path.basename(input_image)
        results = self.zl_model(input_image)

        ratio_x = None
        ratio_y = None
        zl_lengths = []

        # 先找 SK 计算比例
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = class_names[cls_id]
                if cls_name == 'SK':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    width_px = abs(x2 - x1)
                    height_px = abs(y2 - y1)
                    ratio_x = sk_long_cm / width_px
                    ratio_y = sk_short_cm / height_px
                    break

        # 只处理 ZL
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = class_names[cls_id]
                if cls_name == 'ZL':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    width_px = abs(x2 - x1)
                    height_px = abs(y2 - y1)
                    if ratio_x is not None and ratio_y is not None:
                        real_width = round(width_px * ratio_x, 2)
                        real_height = round(height_px * ratio_y, 2)
                    else:
                        real_width = real_height = -1

                    zl_lengths.append({
                        'pixel_width': width_px,
                        'pixel_height': height_px,
                        'real_width_cm': real_width,
                        'real_height_cm': real_height
                    })

                    logger.info(f"ZL像素长度=({width_px},{height_px})，真实长度=({real_width},{real_height})cm")

        return zl_lengths


# 全局分析器实例
analyzer = None


def get_analyzer():
    """获取分析器实例（单例模式）"""
    global analyzer
    if analyzer is None:
        analyzer = IntegratedYMAnalyzer(device='cpu')
    return analyzer


@ym_all_analyzer_ns.route('/analyze', methods=['POST'])
class YMLastAnalyze(Resource):
    @ym_all_analyzer_ns.doc(
        description='上传图片进行YM形状分析（完整版）',
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

                try:
                    # 获取分析器并运行分析
                    analyzer_instance = get_analyzer()
                    zl_lengths = analyzer_instance.calc_zl_length_single(upload_path)
                    analysis_results = analyzer_instance.run_complete_analysis_pipeline(
                        image_path=upload_path,
                        result_dir=result_dir
                    )

                    if not analysis_results:
                        return make_response(jsonify({
                            "code": 400,
                            "message": "未检测到YM区域或分析失败",
                            "data": None
                        }), 400)

                    # 新增：process_single_image处理
                    process_single_image_result = analyzer_instance.process_single_image(upload_path, result_dir)
                    # 路径转为相对static
                    def to_static_url(abs_path):
                        if abs_path is None:
                            return None
                        static_dir = os.path.join(app_root, 'static')
                        rel_path = os.path.relpath(abs_path, static_dir)
                        return f"/static/{rel_path.replace(os.sep, '/')}"
                    process_single_image_urls = None
                    if process_single_image_result:
                        img_keys = {'gs_only_image', 'unet_result', 'highlight', 'center_part'}
                        process_single_image_urls = {
                            k: to_static_url(v) if k in img_keys and v is not None else v
                            for k, v in process_single_image_result.items()
                        }

                    # 处理结果，返回图片路径
                    results = []
                    for i, result in enumerate(analysis_results):
                        # 转换为相对路径 - 相对于static目录
                        static_dir = os.path.join(app_root, 'static')
                        mask_relative_path = os.path.relpath(result['mask_path'], static_dir)
                        overlay_relative_path = os.path.relpath(result['overlay_path'], static_dir)
                        analysis_relative_path = os.path.relpath(result['analysis_path'], static_dir) if result[
                            'analysis_path'] else None
                        crop_relative_path = os.path.relpath(result['crop_path'], static_dir) if result[
                            'crop_path'] else None
                        upload_relative_path = os.path.relpath(upload_path, static_dir) if upload_path else None

                        # 格式化为URL
                        mask_url = f"/static/{mask_relative_path.replace(os.sep, '/')}"
                        overlay_url = f"/static/{overlay_relative_path.replace(os.sep, '/')}"
                        analysis_url = f"/static/{analysis_relative_path.replace(os.sep, '/')}" if analysis_relative_path else None
                        crop_url = f"/static/{crop_relative_path.replace(os.sep, '/')}" if crop_relative_path else None
                        upload_url = f"/static/{upload_relative_path.replace(os.sep, '/')}" if upload_relative_path else None

                        result_data = {
                            "shape_type": result['shape_type'],
                            "midline_widths": [float(w) for w in result['midline_widths']],
                            "image": {
                                "mask": mask_url,
                                "overlay": overlay_url,
                                "analysis": analysis_url
                            },
                            "zl_lengths": zl_lengths,
                            "process_single_image": process_single_image_urls
                        }

                        # 计算比值
                        ratios = {}
                        if len(result['midline_widths']) == 5 and result['midline_widths'][2] != 0 and \
                                result['midline_widths'][4] != 0:
                            ratios = {
                                "ratio_1_3": float(result['midline_widths'][0] / result['midline_widths'][2]),
                                "ratio_3_5": float(result['midline_widths'][2] / result['midline_widths'][4]),
                                "ratio_1_5": float(result['midline_widths'][0] / result['midline_widths'][4])
                            }
                        result_data["ratios"] = ratios

                        results.append(result_data)

                        # 入库操作
                        try:
                            user_id = request.headers.get('token')
                            if user_id:
                                widths = result['midline_widths']
                                history = YmHistory(
                                    user_id=user_id,
                                    upload_path=upload_url,
                                    cropped_ym_path=crop_url,
                                    mask_path=mask_url,
                                    overlay_path=overlay_url,
                                    analysis_path=analysis_url,
                                    shape_type=result['shape_type'],
                                    width1=widths[0] if len(widths) > 0 else None,
                                    width2=widths[1] if len(widths) > 1 else None,
                                    width3=widths[2] if len(widths) > 2 else None,
                                    width4=widths[3] if len(widths) > 3 else None,
                                    width5=widths[4] if len(widths) > 4 else None,
                                    ratio_1_3=ratios.get('ratio_1_3'),
                                    ratio_3_5=ratios.get('ratio_3_5'),
                                    ratio_1_5=ratios.get('ratio_1_5'),
                                    created_time=datetime.now()
                                )
                                db.session.add(history)
                                db.session.commit()
                        except Exception as e:
                            db.session.rollback()
                            logger.error(f"YM历史记录保存失败: {str(e)}")

                    return make_response(jsonify({
                        "code": 200,
                        "message": "分析完成",
                        "data": {
                            "results": results
                        }
                    }), 200)

                except Exception as e:
                    logger.error(f"分析过程中出错: {str(e)}")
                    return make_response(jsonify({
                        "code": 500,
                        "message": f"分析过程中出错: {str(e)}",
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
api.add_namespace(ym_all_analyzer_ns)
