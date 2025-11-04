import os
import cv2
import numpy as np
from PIL import Image
import scipy.spatial
import scipy.ndimage
import torch
import torchvision.transforms as standard_transforms
from collections import Counter

from app import api, db
from flask_restx import Namespace, Resource
from flask import request, jsonify, Blueprint, make_response
import logging
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
from app.models.corn.specific.earRows.models import build_model
from app.utils.path_utils import convert_to_url_path

logger = logging.getLogger(__name__)

# 创建蓝图和命名空间
corn_ear_row_bp = Blueprint('corn_ear_row', __name__)
corn_ear_row_ns = Namespace('corn_ear_row', description='玉米穗行数量检测API')

# 获取应用根目录的绝对路径
app_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
# 配置上传目录
UPLOAD_FOLDER = os.path.join(app_root, 'static', 'images/corn_ear_row/uploads')
RESULT_FOLDER = os.path.join(app_root, 'static', 'images/corn_ear_row/results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# 模型路径配置
P2P_WEIGHT_PATH = r'static/models/specific/earRow/best_mae.pth'

# 检测参数
DEFAULT_THRESHOLD = 0.5
DEFAULT_KERNEL_SIZE = (25, 1)
DEFAULT_MERGE_THRESHOLD_RATIO = 1.5

# 确保目录存在
for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 全局模型缓存
_MODEL_CACHE = None


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------- 工具函数 --------------------- #
def gaussian_filter_density(gt):
    """改进的密度图生成"""
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    h, w = gt.shape

    if gt_count > 10:
        base_sigma = min(h, w) / 100.0
        sigma = max(2.0, min(base_sigma, 4.0))
    else:
        sigma = max(3.0, min(h, w) / 80.0)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        density += scipy.ndimage.gaussian_filter(pt2d, sigma, mode='constant')

    return density


def generate_block_count_map(density_map, kernel_size):
    """生成块计数图"""
    h, w = density_map.shape
    kh, kw = kernel_size

    kernel = np.ones(kernel_size, dtype=np.float32)
    pad_h = kh // 2
    pad_w = kw // 2
    padded_density = np.pad(density_map, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    downsampled_density = cv2.filter2D(padded_density, -1, kernel)[pad_h:pad_h + h, pad_w:pad_w + w]

    non_zero_values = downsampled_density[downsampled_density > 0]
    if len(non_zero_values) > 0:
        mean_value = np.mean(non_zero_values)
        std_value = np.std(non_zero_values)
        dynamic_threshold = max(mean_value * 0.3, mean_value - std_value * 0.5)
    else:
        dynamic_threshold = np.mean(downsampled_density)

    block_map = np.where(downsampled_density >= dynamic_threshold, downsampled_density, 0)
    return block_map


def localize_ear_rows_in_row(Mi, ti):
    """对一行进行聚类定位"""
    Cs = []
    Vm = []
    width = len(Mi)

    for i in range(width):
        if Mi[i] == 0:
            continue

        if not Cs:
            Cs.append([i])
            Vm.append(i)
            continue

        distances = [abs(v - i) for v in Vm]
        min_idx = np.argmin(distances)

        if distances[min_idx] > ti:
            Cs.append([i])
            Vm.append(i)
        else:
            Cs[min_idx].append(i)
            cluster_indices = Cs[min_idx]
            cluster_values = Mi[cluster_indices]
            new_center = np.sum(cluster_indices * cluster_values) / np.sum(cluster_values)
            Vm[min_idx] = new_center

    return Cs, Vm


def calculate_min_distance_to_cluster(current_center, existing_cluster):
    """计算当前点与簇中最近点的距离"""
    if not existing_cluster:
        return float('inf')
    distances = [abs(point_x - current_center) for _, point_x in existing_cluster]
    return min(distances)


def check_continuity_constraint(current_row, existing_cluster, max_gap=5):
    """检查连续性约束"""
    if not existing_cluster:
        return True
    existing_rows = [row for row, _ in existing_cluster]
    min_row_gap = min(abs(row - current_row) for row in existing_rows)
    return min_row_gap <= max_gap


def trace_ear_rows(all_centers, interval_threshold, merge_threshold_ratio):
    """追踪穗行"""
    Cr = []
    cluster_mean = []
    merge_threshold = interval_threshold * merge_threshold_ratio

    for row_num, centers in all_centers:
        if not centers:
            continue

        if not Cr:
            for center in centers:
                Cr.append([(row_num, center)])
                cluster_mean.append(center)
            continue

        for current_center in centers:
            best_cluster_idx = -1
            best_distance = float('inf')

            # 遍历现有簇，找到最佳匹配
            for cluster_idx, cluster in enumerate(Cr):
                if not cluster:
                    continue

                # 计算与簇中最近点的距离
                min_distance = calculate_min_distance_to_cluster(current_center, cluster)

                # 距离约束：超过合并阈值则跳过
                if min_distance > merge_threshold:
                    continue

                # 连续性约束：行间距不能太大
                if not check_continuity_constraint(row_num, cluster, max_gap=5):  # 使用 row_num
                    continue

                # 更新最佳匹配
                if min_distance < best_distance:
                    best_distance = min_distance
                    best_cluster_idx = cluster_idx

            # 连接到最佳簇或创建新簇
            if best_cluster_idx != -1:
                Cr[best_cluster_idx].append((row_num, current_center))
                cluster_mean[best_cluster_idx] = np.mean([point[1] for point in Cr[best_cluster_idx]])
            else:
                Cr.append([(row_num, current_center)])
                cluster_mean.append(current_center)

    # 计算最终簇中心
    final_cluster_mean = []
    for r_cluster in Cr:
        if r_cluster:
            x_coords = [point[1] for point in r_cluster]
            final_cluster_mean.append(np.mean(x_coords))
        else:
            final_cluster_mean.append(None)

    return Cr, final_cluster_mean


def generate_gradient_colors(n_colors):
    """生成渐变色"""
    if n_colors <= 1:
        return [[0, 255, 0]]

    colors = []
    start_hue = 0
    end_hue = 300
    saturation = 0.9
    value = 1.0

    for i in range(n_colors):
        if n_colors == 1:
            hue = start_hue
        else:
            hue = start_hue + (end_hue - start_hue) * i / (n_colors - 1)

        hue = hue % 360

        if n_colors > 2:
            hue_offset = (i * 20) % 60
            hue = (hue + hue_offset) % 360

        h = hue / 360.0
        s = saturation
        v = value

        if s == 0:
            r = g = b = v
        else:
            h = h * 6
            i_h = int(h)
            f = h - i_h
            p = v * (1 - s)
            q = v * (1 - s * f)
            t = v * (1 - s * (1 - f))

            if i_h == 0:
                r, g, b = v, t, p
            elif i_h == 1:
                r, g, b = q, v, p
            elif i_h == 2:
                r, g, b = p, v, t
            elif i_h == 3:
                r, g, b = p, q, v
            elif i_h == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q

        colors.append([int(b * 255), int(g * 255), int(r * 255)])

    return colors


def visualize_tracking_result(img, Cr, final_cluster_mean):
    """可视化追踪结果"""
    img_tracking = img.copy()

    if not Cr:
        return img_tracking

    n_clusters = len(Cr)
    cluster_colors = generate_gradient_colors(n_clusters)

    for cluster_idx, cluster in enumerate(Cr):
        if not cluster:
            continue

        sorted_cluster = sorted(cluster, key=lambda x: x[0])
        color = cluster_colors[cluster_idx]

        for i in range(len(sorted_cluster) - 1):
            row1, x1 = sorted_cluster[i]
            row2, x2 = sorted_cluster[i + 1]
            pt1 = (int(round(x1)), int(row1))
            pt2 = (int(round(x2)), int(row2))

            if (0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0] and
                    0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]):
                cv2.line(img_tracking, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)

        for row, x in sorted_cluster:
            pt = (int(round(x)), int(row))
            if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
                cv2.circle(img_tracking, pt, 4, color, -1)
                cv2.circle(img_tracking, pt, 4, (255, 255, 255), 1)

    # 添加图例
    legend_x = 20
    legend_y = 30
    legend_spacing = 25
    max_per_column = 8
    column_width = 80

    for i in range(n_clusters):
        column = i // max_per_column
        row_in_column = i % max_per_column
        current_x = legend_x + column * column_width
        current_y = legend_y + row_in_column * legend_spacing

        color = cluster_colors[i]
        cv2.rectangle(img_tracking, (current_x, current_y - 10),
                      (current_x + 20, current_y + 10), color, -1)
        cv2.rectangle(img_tracking, (current_x, current_y - 10),
                      (current_x + 20, current_y + 10), (255, 255, 255), 2)
        cv2.putText(img_tracking, f'Row {i + 1}',
                    (current_x + 25, current_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    title_text = f'Tracking Result: {n_clusters} Ear Rows Detected'
    cv2.putText(img_tracking, title_text, (legend_x, legend_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img_tracking, title_text, (legend_x, legend_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    return img_tracking


def select_optimal_even_clusters(all_clusters):
    """选择最优偶数簇数量"""
    cluster_counts = []
    for row_num, Cs in all_clusters:
        cluster_count = len(Cs)
        cluster_counts.append(cluster_count)

    even_cluster_counts = [count for count in cluster_counts if count % 2 == 0 and count > 0]

    if not even_cluster_counts:
        logger.warning("没有检测到偶数簇，使用默认值 2")
        return 2

    count_frequency = Counter(even_cluster_counts)
    most_common_count = count_frequency.most_common(1)[0][0]

    return most_common_count


# --------------------- 核心检测类 --------------------- #
class CornEarRowDetector:
    """玉米穗行数量检测器"""

    def __init__(self, weight_path, threshold=0.5, kernel_size=(25, 1),
                 merge_threshold_ratio=1.5):
        """
        初始化检测器

        Args:
            weight_path: P2PNet模型权重路径
            threshold: 预测阈值
            kernel_size: 卷积核大小
            merge_threshold_ratio: 合并阈值比例
        """
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"P2PNet 权重不存在: {weight_path}")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.merge_threshold_ratio = merge_threshold_ratio

        # 加载P2PNet模型
        try:
            class ModelArgs:
                def __init__(self):
                    self.backbone = 'vgg16_bn'
                    self.row = 2
                    self.line = 2

            args = ModelArgs()
            self.model = build_model(args)
            self.model.to(self.device)

            checkpoint = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.model.eval()

            logger.info(f"P2PNet 加载完成 | 设备: {self.device}")

        except Exception as e:
            raise ValueError(f"P2PNet模型加载失败: {str(e)}")

        # 图像预处理
        self.transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]),
        ])

    def predict_points(self, img_path):
        """使用P2PNet预测点坐标"""
        img_raw = Image.open(img_path).convert('RGB')
        width, height = img_raw.size

        # 调整图像大小
        min_side = min(width, height)
        max_side = max(width, height)

        if min_side < 512:
            scale = 512 / min_side
            new_width = int(width * scale)
            new_height = int(height * scale)
        elif max_side > 1024:
            scale = 1024 / max_side
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            new_width, new_height = width, height

        new_width = (new_width // 32) * 32
        new_height = (new_height // 32) * 32

        img_raw = img_raw.resize((new_width, new_height), Image.LANCZOS)

        # 预处理
        img = self.transform(img_raw)
        samples = torch.Tensor(img).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(samples)
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            outputs_points = outputs['pred_points'][0]

        # 过滤结果
        points = outputs_points[outputs_scores > self.threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > self.threshold).sum())

        logger.info(f"P2PNet预测点数量: {predict_cnt}")
        return points, img_raw

    def detect(self, image_path, result_save_path=None):
        """
        检测穗行数量

        Args:
            image_path: 输入图片路径
            result_save_path: 结果图保存路径（可选）

        Returns:
            {
                'success': True/False,
                'msg': 消息,
                'data': {
                    'number_of_ears': 穗行数量
                },
                'result_path': 结果图路径
            }
        """
        try:
            # 步骤1: P2PNet预测点
            points, img_raw = self.predict_points(image_path)

            # 转换为OpenCV格式
            img = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]

            # 步骤2: 生成点图
            point_map = np.zeros((h, w), dtype=np.float32)
            for p in points:
                x = min(w - 1, max(0, int(round(p[0]))))
                y = min(h - 1, max(0, int(round(p[1]))))
                point_map[y, x] = 1

            # 步骤3: 生成密度图
            density_map = gaussian_filter_density(point_map)

            # 步骤4: 生成块计数图
            block_map = generate_block_count_map(density_map, self.kernel_size)

            # 步骤5: 计算间隔阈值
            interval_threshold = self._calculate_interval_threshold(block_map)

            # 步骤6: 定位簇
            all_clusters, all_centers = self._localize_clusters(block_map, interval_threshold, h)

            # 步骤7: 追踪穗行
            Cr, final_cluster_mean = trace_ear_rows(all_centers, interval_threshold,
                                                    self.merge_threshold_ratio)

            # 步骤8: 选择最优偶数簇数量
            optimal_even_count = select_optimal_even_clusters(all_clusters)

            # 步骤9: 可视化结果
            img_result = visualize_tracking_result(img, Cr, final_cluster_mean)

            # 保存结果图
            result_saved = False
            if result_save_path:
                try:
                    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
                    cv2.imwrite(result_save_path, img_result)
                    result_saved = True
                    logger.info(f"结果图已保存: {result_save_path}")
                except Exception as e:
                    logger.error(f"结果图保存失败: {str(e)}")

            return {
                'success': True,
                'msg': '检测成功',
                'data': {
                    'number_of_ears': int(optimal_even_count)
                },
                'result_path': result_save_path if result_saved else None
            }

        except Exception as e:
            logger.error(f"检测失败: {str(e)}", exc_info=True)
            return {
                'success': False,
                'msg': f'检测失败: {str(e)}',
                'data': {'number_of_ears': 0},
                'result_path': None
            }

    def _calculate_interval_threshold(self, block_map):
        """计算间隔阈值"""
        non_zero_rows = np.where(np.any(block_map > 0, axis=1))[0]

        if len(non_zero_rows) == 0:
            return 20

        min_row = non_zero_rows.min()
        max_row = non_zero_rows.max()
        density_center_row = min_row + (max_row - min_row) // 2
        row_data = block_map[density_center_row, :]

        column_starts = []
        current_column_start = None

        for col_idx in range(len(row_data)):
            if row_data[col_idx] > 0:
                if current_column_start is None:
                    current_column_start = col_idx
            else:
                if current_column_start is not None:
                    column_starts.append(current_column_start)
                    current_column_start = None

        if current_column_start is not None:
            column_starts.append(current_column_start)

        if len(column_starts) < 2:
            return 20

        distances = []
        for i in range(1, len(column_starts)):
            dist = column_starts[i] - column_starts[i - 1]
            if dist > 0:
                distances.append(dist)

        if not distances:
            return 20

        avg_distance = np.mean(distances)
        calculated_threshold = max(10, int(avg_distance * 0.5))

        return calculated_threshold

    def _localize_clusters(self, block_map, interval_threshold, h):
        """定位簇"""
        non_zero_rows = np.where(np.any(block_map > 0, axis=1))[0]
        if len(non_zero_rows) == 0:
            start_h = h // 3
            end_h = 2 * h // 3
        else:
            min_row = non_zero_rows.min()
            max_row = non_zero_rows.max()
            total_range = max_row - min_row
            start_h = min_row + total_range // 3
            end_h = min_row + 2 * total_range // 3

        block_map_center = block_map[start_h:end_h, :]

        all_clusters = []
        all_centers = []

        for row_idx in range(block_map_center.shape[0]):
            Mi = block_map_center[row_idx, :]
            Cs, Vm = localize_ear_rows_in_row(Mi, interval_threshold)
            all_clusters.append((start_h + row_idx, Cs))
            all_centers.append((start_h + row_idx, Vm))

        return all_clusters, all_centers


def get_detector():
    """获取或创建检测器单例"""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        try:
            _MODEL_CACHE = CornEarRowDetector(
                weight_path=P2P_WEIGHT_PATH,
                threshold=DEFAULT_THRESHOLD,
                kernel_size=DEFAULT_KERNEL_SIZE,
                merge_threshold_ratio=DEFAULT_MERGE_THRESHOLD_RATIO
            )
        except Exception as e:
            logger.error(f"初始化检测器失败: {str(e)}")
            raise
    return _MODEL_CACHE


# --------------------- Flask-RESTX 资源 --------------------- #
@corn_ear_row_ns.route('/detect', methods=['POST'])
class CornEarRowDetect(Resource):
    @corn_ear_row_ns.doc(
        description='上传玉米图片，检测穗行数量',
        responses={
            200: '检测成功',
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

            # 准备结果输出路径
            result_path = os.path.join(RESULT_FOLDER, f"{base_name}_result.png")

            try:
                # === 核心检测流程 === #
                detector = get_detector()
                result = detector.detect(upload_path, result_save_path=result_path)

                if not result['success']:
                    return make_response(jsonify({
                        "code": 400,
                        "message": result['msg'],
                        "data": result['data']
                    }), 400)

                # 转换为URL路径
                upload_url = convert_to_url_path(upload_path, app_root)

                result_url = None
                if result.get('result_path') and os.path.exists(result['result_path']):
                    result_url = convert_to_url_path(result['result_path'], app_root)

                # # 保存历史记录
                # try:
                #     user_id = request.headers.get('token')
                #     if user_id:
                #         history = CornEarRowHistory(
                #             user_id=user_id,
                #             upload_path=upload_url,
                #             result_path=result_url,
                #             number_of_ears=result['data']['number_of_ears'],
                #             created_time=datetime.now()
                #         )
                #         db.session.add(history)
                #         db.session.commit()
                #         logger.info(f"历史记录已保存，ID: {history.id}")
                # except Exception as e:
                #     db.session.rollback()
                #     logger.error(f"历史记录保存失败: {str(e)}")
                #
                # 返回成功结果
                return make_response(jsonify({
                    "code": 200,
                    "message": "检测成功",
                    "data": {
                        "number_of_ears": result['data']['number_of_ears'],
                        "original_image": upload_url,
                        "result_image": result_url
                    }
                }), 200)

            except ValueError as ve:
                logger.error(f"检测失败: {str(ve)}")
                return make_response(jsonify({
                    "code": 400,
                    "message": str(ve)
                }), 400)

            except Exception as e:
                logger.error(f"图像检测过程中出错: {str(e)}", exc_info=True)
                return make_response(jsonify({
                    "code": 500,
                    "message": f"图像检测过程中出错: {str(e)}"
                }), 500)

        except Exception as e:
            logger.error(f"服务器内部错误: {str(e)}", exc_info=True)
            return make_response(jsonify({
                "code": 500,
                "message": f"服务器内部错误: {str(e)}",
                "data": None
            }), 500)


# 注册命名空间
api.add_namespace(corn_ear_row_ns)