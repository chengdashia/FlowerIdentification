import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import base64
import io
from flask import Blueprint, request, jsonify, make_response
from app.utils.model_loader import load_ym_models
from flask_restx import Namespace, Resource
from app import api
import logging

logger = logging.getLogger(__name__)

# 创建蓝图和命名空间
ym_analyzer_bp = Blueprint('ym_analyzer', __name__)
ym_analyzer_ns = Namespace('ym_analyzer', description='YM shape analysis API 使用base64编码的图片进行分析')

# 加载模型
yolo_model, unet_model, zl_model, ym_model = load_ym_models()


def decode_base64_to_pil(base64_str):
    """base64 -> PIL Image"""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[-1]
        img_bytes = base64.b64decode(base64_str)
        img_pil = Image.open(io.BytesIO(img_bytes))
        
        # 修复图像方向问题
        try:
            # 获取EXIF信息中的方向
            exif = img_pil._getexif()
            if exif is not None:
                orientation = exif.get(274)  # 274是方向标签的ID
                if orientation is not None:
                    # 根据方向信息旋转图像
                    if orientation == 3:
                        img_pil = img_pil.rotate(180, expand=True)
                    elif orientation == 6:
                        img_pil = img_pil.rotate(270, expand=True)
                    elif orientation == 8:
                        img_pil = img_pil.rotate(90, expand=True)
        except Exception as e:
            logger.warning(f"处理图像方向时出错: {e}")
        
        return img_pil
    except Exception as e:
        logger.error(f"[decode_base64_to_pil] 失败: {e}")
        return None


def encode_image_to_base64(img_array, format='JPEG'):
    """numpy array -> base64"""
    try:
        if isinstance(img_array, np.ndarray):
            img_pil = Image.fromarray(img_array)
        else:
            img_pil = img_array

        buffer = io.BytesIO()
        img_pil.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_str
    except Exception as e:
        logger.error(f"[encode_image_to_base64] 失败: {e}")
        return None


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

    # 添加调试信息
    logger.info(f"掩码边界: min_row={min_row}, max_row={max_row}, 总高度={max_row-min_row+1}")

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

            # 添加调试信息
            logger.info(f"段{segment+1}: 行范围[{segment_start_row}-{segment_end_row}], 中位线行={midline_row}, 宽度={mask_pixels_in_row}")

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

        # 使用默认字体，不需要加载外部字体
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


def detect_and_crop_ym(image_pil):
    """检测并裁剪YM区域"""
    try:
        # 直接从内存处理图像，不需要临时文件
        img_np = np.array(image_pil.convert('RGB'))
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 使用YOLO模型检测
        logger.info(f"开始YOLO检测，模型类型: {type(yolo_model)}")
        results = yolo_model.predict(img_np)
        logger.info(f"YOLO检测完成，结果数量: {len(results)}")

        cropped_images = []

        for i, result in enumerate(results):
            logger.info(f"处理第{i + 1}个检测结果")

            # 对于分割模型，我们需要使用masks而不是boxes
            if hasattr(result, 'masks') and result.masks is not None:
                logger.info(f"第{i + 1}个结果分割掩码数量: {len(result.masks)}")

                for j, mask in enumerate(result.masks):
                    # 获取类别ID
                    if hasattr(mask, 'cls'):
                        cls_id = int(mask.cls.cpu().item())
                    else:
                        # 如果mask没有cls属性，尝试从result获取
                        cls_id = int(result.boxes[j].cls.cpu().item()) if j < len(result.boxes) else 0

                    class_name = yolo_model.names[cls_id]
                    logger.info(f"检测到类别: {class_name} (ID: {cls_id})")

                    if class_name == 'YM':
                        # 获取掩码数据
                        mask_data = mask.data.cpu().numpy()[0]  # 获取掩码数组

                        # 将掩码调整到原始图像大小
                        mask_data = cv2.resize(mask_data, (img_np.shape[1], img_np.shape[0]))

                        # 找到掩码的边界框
                        contours, _ = cv2.findContours((mask_data > 0.5).astype(np.uint8),
                                                       cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)

                        if contours:
                            # 获取最大轮廓的边界框
                            cnt = max(contours, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(cnt)

                            # 裁剪图像
                            crop = img_np[y:y + h, x:x + w]

                            if crop.size == 0:
                                logger.error("裁剪区域为空")
                                continue

                            # 转换为PIL图像
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            crop_pil = Image.fromarray(crop_rgb)
                            cropped_images.append(crop_pil)
                            logger.info(f"成功裁剪YM区域 {len(cropped_images)}")

            # 如果没有掩码但有边界框，也尝试使用边界框
            elif hasattr(result, 'boxes') and result.boxes is not None:
                logger.info(f"第{i + 1}个结果检测框数量: {len(result.boxes)}")
                for j, box in enumerate(result.boxes):
                    cls_id = int(box.cls.cpu().item())
                    class_name = yolo_model.names[cls_id]
                    logger.info(f"检测到类别: {class_name} (ID: {cls_id})")

                    if class_name == 'YM':
                        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())
                        crop = img_np[y1:y2, x1:x2]

                        if crop.size == 0:
                            logger.error("裁剪区域为空")
                            continue

                        # 转换为PIL图像
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        crop_pil = Image.fromarray(crop_rgb)
                        cropped_images.append(crop_pil)
                        logger.info(f"成功裁剪YM区域 {len(cropped_images)}")

        logger.info(f"总共裁剪了 {len(cropped_images)} 个YM区域")
        return cropped_images

    except Exception as e:
        logger.error(f"detect_and_crop_ym 函数出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_mask(image_pil):
    """对单张图像进行掩码预测"""
    # 确保图像是RGB模式
    image_pil = image_pil.convert('RGB')
    original_size = image_pil.size

    # 添加调试信息
    logger.info(f"原始图像尺寸: {original_size}")

    # 预处理图像
    image_tensor = transform(image_pil).unsqueeze(0)

    # 预测
    with torch.no_grad():
        prediction = unet_model(image_tensor)
        if prediction.dim() == 4:
            prediction = prediction.squeeze(1)
        prediction = torch.sigmoid(prediction)
        prediction = prediction.squeeze().cpu().numpy()

    # 二值化掩码
    binary_mask = (prediction > 0.5).astype(np.uint8) * 255

    # 调整回原始尺寸
    mask_pil = Image.fromarray(binary_mask)
    mask_pil = mask_pil.resize((original_size[0], original_size[1]), Image.NEAREST)
    mask_array = np.array(mask_pil)

    # 添加调试信息
    logger.info(f"掩码尺寸: {mask_array.shape}")

    return mask_array


def create_overlay_image(original_pil, mask_array):
    """创建叠加效果图像"""
    # 转换为numpy数组
    original_array = np.array(original_pil)

    # 确保掩码尺寸匹配
    if mask_array.shape[:2] != original_array.shape[:2]:
        mask_pil = Image.fromarray(mask_array)
        mask_pil = mask_pil.resize((original_array.shape[1], original_array.shape[0]), Image.NEAREST)
        mask_array = np.array(mask_pil)

    # 创建叠加效果
    color_mask = np.zeros_like(original_array)
    color_mask[mask_array > 0] = [255, 165, 0]  # 橙色掩码
    cv_overlay = cv2.addWeighted(original_array, 1, color_mask, 0.7, 0)

    return cv_overlay


def process_image(image_pil):
    """处理单个图像并返回分析结果"""
    try:
        # 预测掩码
        mask_array = predict_mask(image_pil)

        # 创建叠加效果
        overlay_image = create_overlay_image(image_pil, mask_array)

        # 进行形状分析
        colored_mask, shape_result, midline_widths = split_mask_horizontally(
            mask_array,
            add_midlines=True,
            midline_thickness=2,
            analyze_shape=True
        )

        # 计算比值
        ratios = {}
        if len(midline_widths) == 5 and midline_widths[2] != 0 and midline_widths[4] != 0:
            ratios = {
                "ratio_1_3": float(midline_widths[0] / midline_widths[2]),
                "ratio_3_5": float(midline_widths[2] / midline_widths[4]),
                "ratio_1_5": float(midline_widths[0] / midline_widths[4])
            }
            
            # 添加调试信息
            logger.info(f"中位线宽度: {midline_widths}")
            logger.info(f"比例计算: 1/3={ratios['ratio_1_3']:.3f}, 3/5={ratios['ratio_3_5']:.3f}, 1/5={ratios['ratio_1_5']:.3f}")

        # 转换为base64
        original_base64 = encode_image_to_base64(image_pil, format='PNG')
        mask_base64 = encode_image_to_base64(mask_array, format='PNG')
        overlay_base64 = encode_image_to_base64(overlay_image, format='PNG')
        analysis_base64 = encode_image_to_base64(colored_mask, format='PNG')

        # 创建结果
        result = {
            "shape_type": shape_result,
            "midline_widths": [float(w) for w in midline_widths],
            "ratios": ratios,
            "images": {
                "mask": mask_base64,
                "overlay": overlay_base64,
                "analysis": analysis_base64
            }
        }

        return result

    except Exception as e:
        logger.error(f"处理图像时出错: {str(e)}")
        return None


def analyze_ym_image(image_pil):
    """完整的YM分析流程"""
    try:
        # 检测并裁剪YM区域
        cropped_images = detect_and_crop_ym(image_pil)

        if not cropped_images:
            return {
                "code": 400,
                "message": "未检测到YM区域",
                "data": None
            }

        # 处理每个裁剪的图像
        results = []
        for i, crop_pil in enumerate(cropped_images):
            result = process_image(crop_pil)
            if result:
                result["ym_index"] = i + 1
                results.append(result)

        if not results:
            return {
                "code": 500,
                "message": "图像处理失败",
                "data": None
            }

        return {
            "code": 200,
            "message": "处理成功",
            "data": {
                "ym_count": len(results),
                "results": results
            }
        }

    except Exception as e:
        return {
            "code": 500,
            "message": f"服务器内部错误: {str(e)}",
            "data": None
        }


@ym_analyzer_ns.route('/analyze', methods=['POST'])
class YMAnalyze(Resource):
    @ym_analyzer_ns.doc(
        description='上传图片进行YM形状分析',
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

            # 分析图像
            result = analyze_ym_image(pil_image)

            # 返回结果
            return make_response(jsonify(result), result["code"])

        except Exception as e:
            return make_response(jsonify({
                "code": 500,
                "message": f"服务器内部错误: {str(e)}",
                "data": None
            }), 500)


# 注册命名空间
api.add_namespace(ym_analyzer_ns)