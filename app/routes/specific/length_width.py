import os
import cv2
import numpy as np
from app import api
from flask_restx import Namespace, Resource
from flask import request, jsonify, Blueprint, make_response
import logging
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

from app.models.corn.ultralytics import YOLO
from app.utils.path_utils import convert_to_url_path

logger = logging.getLogger(__name__)

# 创建蓝图和命名空间
corn_lw_bp = Blueprint('corn_lw', __name__)
corn_lw_ns = Namespace('corn_lw', description='玉米宽度长度测量API')

# 获取应用根目录的绝对路径
app_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# 配置上传目录
UPLOAD_FOLDER = os.path.join(app_root, 'static', 'images/corn_ear_row/uploads')
RESULT_FOLDER = os.path.join(app_root, 'static', 'images/corn_ear_row/results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# 模型路径配置
# YOLO_MODEL_PATH = r'static/models/specific/grade/best.pt'

YOLO_MODEL_PATH = os.path.join(app_root, 'static/models/specific/grade/best.pt')

# 默认测量参数
DEFAULT_CONF_THRES = 0.25
DEFAULT_IOU_THRES = 0.5
DEFAULT_IMG_SIZE = 1024
DEFAULT_RETINA_MASKS = True
DEFAULT_INSTANCE_CONF_MIN = 0.85

# 标定参数（等比例标定）
DEFAULT_REF_W_PX = 2635.54
DEFAULT_REF_H_PX = 2635.54
DEFAULT_REF_W_CM = 20.7
DEFAULT_REF_H_CM = 20.7
DEFAULT_NDIGITS = 4

# 确保目录存在
for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 全局模型缓存
_MODEL_CACHE = None


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------- 核心测量类 --------------------- #
class CornWidthMeasurer:
    """玉米宽度长度测量器"""

    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.5,
                 imgsz=1024, retina_masks=True, instance_conf_min=0.85,
                 ref_w_px=2635.54, ref_h_px=2635.54,
                 ref_w_cm=20.7, ref_h_cm=20.7, ndigits=4):
        """
        初始化测量器

        Args:
            model_path: YOLO模型路径
            conf_thres: 置信度阈值
            iou_thres: IOU阈值
            imgsz: 图像大小
            retina_masks: 是否使用retina masks
            instance_conf_min: 实例最小置信度
            ref_w_px, ref_h_px: 参考宽度高度（像素）
            ref_w_cm, ref_h_cm: 参考宽度高度（厘米）
            ndigits: 保留小数位数
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.retina_masks = retina_masks
        self.instance_conf_min = instance_conf_min
        self.ndigits = ndigits

        # 标定参数
        try:
            self.CM_PER_PX_X = float(ref_w_cm) / float(ref_w_px)
            self.CM_PER_PX_Y = float(ref_h_cm) / float(ref_h_px)
        except Exception as e:
            raise ValueError(f"标定参数错误: {str(e)}")

        # 加载YOLO模型
        try:
            self.model = YOLO(model_path)

            # 查找玉米类别ID
            self.ym_class_id = None
            for class_id, class_name in self.model.names.items():
                if str(class_name).strip().lower() == "ym":
                    self.ym_class_id = int(class_id)
                    break

            if self.ym_class_id is None:
                raise ValueError("模型类别中未找到 'ym'")

            logger.info(f"测量器初始化完成，玉米类别ID: {self.ym_class_id}")
        except Exception as e:
            raise ValueError(f"模型加载失败: {str(e)}")

    def _cm_per_px(self, vec2):
        """计算向量的厘米/像素比例"""
        vx, vy = float(vec2[0]), float(vec2[1])
        return float(((vx * self.CM_PER_PX_X) ** 2 + (vy * self.CM_PER_PX_Y) ** 2) ** 0.5)

    def _pca_axis(self, mask_bin):
        """
        PCA计算主轴
        返回: (中心点, 主轴向量, 副轴向量, t_min, t_max, 长度像素, (X, Y, t))
        """
        ys, xs = np.where(mask_bin > 0)
        if len(xs) < 20:
            return None

        X = xs.astype(np.float64)
        Y = ys.astype(np.float64)
        cx, cy = X.mean(), Y.mean()

        # 协方差矩阵
        Z = np.stack([X - cx, Y - cy], axis=0)
        C = (Z @ Z.T) / max(1, Z.shape[1])

        # 特征值分解
        _, V = np.linalg.eigh(C)
        v1 = V[:, 1] / (np.linalg.norm(V[:, 1]) + 1e-12)  # 主轴
        v2 = V[:, 0] / (np.linalg.norm(V[:, 0]) + 1e-12)  # 副轴

        # 投影到主轴
        t = v1[0] * (X - cx) + v1[1] * (Y - cy)
        t_min, t_max = float(np.min(t)), float(np.max(t))
        L_px = t_max - t_min

        if not np.isfinite(L_px) or L_px < 5.0:
            return None

        return (cx, cy), v1, v2, t_min, t_max, L_px, (X, Y, t)

    def _center_width(self, center, v1, v2, t_min, t_max, X, Y, t):
        """计算中心宽度"""
        cx, cy = center
        s = v2[0] * (X - cx) + v2[1] * (Y - cy)
        t_half = 0.5 * (t_min + t_max)

        for d in (0.5, 1.0, 1.5, 2.0, 3.0):
            band = np.abs(t - t_half) <= d
            if band.sum() >= 8:
                s_sel = s[band]
                s_min, s_max = float(np.min(s_sel)), float(np.max(s_sel))
                width_px = float(s_max - s_min)
                width_cm = width_px * self._cm_per_px(v2)
                return width_px, width_cm

        return 0.0, 0.0

    def _mid_third_max_width(self, center, v1, v2, t_min, t_max, X, Y, t):
        """计算中间1/3最大宽度"""
        cx, cy = center
        s = v2[0] * (X - cx) + v2[1] * (Y - cy)

        L = t_max - t_min
        t1, t2 = t_min + L / 3.0, t_min + 2.0 * L / 3.0
        mid = (t >= t1) & (t <= t2)

        if mid.sum() < 10:
            return 0.0, 0.0

        t_mid = t[mid] - t_min
        s_mid = s[mid]

        # 按整数t分组
        bin_idx = np.floor(t_mid + 0.5).astype(np.int32)
        s_min_dict, s_max_dict = {}, {}

        for b, sv in zip(bin_idx, s_mid):
            if b not in s_min_dict:
                s_min_dict[b] = sv
                s_max_dict[b] = sv
            else:
                if sv < s_min_dict[b]:
                    s_min_dict[b] = sv
                if sv > s_max_dict[b]:
                    s_max_dict[b] = sv

        if not s_min_dict:
            return 0.0, 0.0

        widths_px = [float(s_max_dict[b] - s_min_dict[b]) for b in s_min_dict.keys()]
        if not widths_px:
            return 0.0, 0.0

        max_px = max(widths_px)
        max_cm = max_px * self._cm_per_px(v2)
        return max_px, max_cm

    def measure(self, image_path, save_vis_path=None):
        """
        测量图像

        Returns:
            {
                'success': True/False,
                'msg': 错误信息,
                'data': {
                    'length': 长度(cm),
                    'center_width': 中心宽度(cm),
                    'max_width_mid_third': 中间1/3最大宽度(cm)
                },
                'vis_path': 可视化图像路径（如果保存）
            }
        """
        # 读取图像
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                return {
                    'success': False,
                    'msg': '图片有误',
                    'data': {'length': 0.0, 'center_width': 0.0, 'max_width_mid_third': 0.0}
                }

            h, w = img.shape[:2]
            if min(h, w) < 8:
                return {
                    'success': False,
                    'msg': '图片尺寸过小',
                    'data': {'length': 0.0, 'center_width': 0.0, 'max_width_mid_third': 0.0}
                }
        except Exception as e:
            logger.error(f"读取图像失败: {str(e)}")
            return {
                'success': False,
                'msg': f'图片有误: {str(e)}',
                'data': {'length': 0.0, 'center_width': 0.0, 'max_width_mid_third': 0.0}
            }

        # YOLO推理
        try:
            results = self.model.predict(
                source=img,
                classes=[self.ym_class_id],
                conf=self.conf_thres,
                iou=self.iou_thres,
                imgsz=self.imgsz,
                retina_masks=self.retina_masks,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"YOLO推理失败: {str(e)}")
            return {
                'success': False,
                'msg': f'推理失败: {str(e)}',
                'data': {'length': 0.0, 'center_width': 0.0, 'max_width_mid_third': 0.0}
            }

        if not results:
            return {
                'success': False,
                'msg': '未识别到玉米',
                'data': {'length': 0.0, 'center_width': 0.0, 'max_width_mid_third': 0.0}
            }

        r = results[0]

        # 选择最佳掩码
        try:
            if r.boxes is None or len(r.boxes) == 0:
                return {
                    'success': False,
                    'msg': '未识别到玉米',
                    'data': {'length': 0.0, 'center_width': 0.0, 'max_width_mid_third': 0.0}
                }

            if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
                return {
                    'success': False,
                    'msg': '未识别到玉米',
                    'data': {'length': 0.0, 'center_width': 0.0, 'max_width_mid_third': 0.0}
                }

            masks = r.masks.data.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((len(masks),), dtype=float)

            # 筛选高置信度
            keep_idx = [i for i, c in enumerate(confs) if float(c) >= self.instance_conf_min]
            if not keep_idx:
                return {
                    'success': False,
                    'msg': '未识别到玉米',
                    'data': {'length': 0.0, 'center_width': 0.0, 'max_width_mid_third': 0.0}
                }

            # 选择面积最大的
            mh, mw = masks.shape[-2], masks.shape[-1]
            best_area, best_mask = -1, None

            for i in keep_idx:
                m = masks[i]
                m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) \
                    if (mh, mw) != (h, w) else m
                mbin = (m_resized > 0.5).astype(np.uint8)
                area = int(mbin.sum())

                if area > best_area:
                    best_area, best_mask = area, mbin

            if best_mask is None or best_area <= 0:
                return {
                    'success': False,
                    'msg': '未识别到玉米',
                    'data': {'length': 0.0, 'center_width': 0.0, 'max_width_mid_third': 0.0}
                }

        except Exception as e:
            logger.error(f"掩码处理失败: {str(e)}")
            return {
                'success': False,
                'msg': f'图片有误: {str(e)}',
                'data': {'length': 0.0, 'center_width': 0.0, 'max_width_mid_third': 0.0}
            }

        # PCA计算主轴
        geo = self._pca_axis(best_mask)
        if geo is None:
            return {
                'success': False,
                'msg': '未识别到玉米',
                'data': {'length': 0.0, 'center_width': 0.0, 'max_width_mid_third': 0.0}
            }

        center, v1, v2, t_min, t_max, L_px, (X_all, Y_all, t_all) = geo

        # 计算长度
        L_cm = float(L_px) * self._cm_per_px(v1)

        # 计算中心宽度
        C_px, C_cm = self._center_width(center, v1, v2, t_min, t_max, X_all, Y_all, t_all)

        # 计算中间1/3最大宽度
        M_px, M_cm = self._mid_third_max_width(center, v1, v2, t_min, t_max, X_all, Y_all, t_all)

        # 可视化
        vis_saved = False
        if save_vis_path:
            try:
                vis = img.copy()

                # 绘制轮廓
                cnts, _ = cv2.findContours(
                    (best_mask * 255).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)

                # 绘制主轴
                P1 = (np.array([center[0], center[1]]) + v1 * t_min).astype(int)
                P2 = (np.array([center[0], center[1]]) + v1 * t_max).astype(int)
                cv2.line(vis, tuple(P1), tuple(P2), (0, 0, 255), 2)

                # 添加文本
                txt = f"L={L_cm:.{self.ndigits}f}cm | Wc={C_cm:.{self.ndigits}f}cm | W1/3max={M_cm:.{self.ndigits}f}cm"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(vis, (10, 10), (10 + tw + 10, 10 + th + 14), (0, 0, 0), -1)
                cv2.putText(vis, txt, (15, 10 + th + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                # 保存
                os.makedirs(os.path.dirname(save_vis_path), exist_ok=True)
                cv2.imwrite(save_vis_path, vis)
                vis_saved = True
                logger.info(f"可视化图像已保存: {save_vis_path}")

            except Exception as e:
                logger.error(f"可视化保存失败: {str(e)}")

        # 返回结果
        return {
            'success': True,
            'msg': '测量成功',
            'data': {
                'length': float(round(L_cm, self.ndigits)),
                'center_width': float(round(C_cm, self.ndigits)),
                'max_width_mid_third': float(round(M_cm, self.ndigits))
            },
            'vis_path': save_vis_path if vis_saved else None
        }


def get_measurer():
    """获取或创建测量器单例"""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        try:
            _MODEL_CACHE = CornWidthMeasurer(
                model_path=YOLO_MODEL_PATH,
                conf_thres=DEFAULT_CONF_THRES,
                iou_thres=DEFAULT_IOU_THRES,
                imgsz=DEFAULT_IMG_SIZE,
                retina_masks=DEFAULT_RETINA_MASKS,
                instance_conf_min=DEFAULT_INSTANCE_CONF_MIN,
                ref_w_px=DEFAULT_REF_W_PX,
                ref_h_px=DEFAULT_REF_H_PX,
                ref_w_cm=DEFAULT_REF_W_CM,
                ref_h_cm=DEFAULT_REF_H_CM,
                ndigits=DEFAULT_NDIGITS
            )
        except Exception as e:
            logger.error(f"初始化测量器失败: {str(e)}")
            raise
    return _MODEL_CACHE


# --------------------- Flask-RESTX 资源 --------------------- #
@corn_lw_ns.route('/measure', methods=['POST'])
class CornWidthMeasure(Resource):
    @corn_lw_ns.doc(
        description='上传玉米图片，测量长度和宽度（厘米）',
        responses={
            200: '测量成功',
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

            # 准备可视化输出路径
            vis_path = os.path.join(RESULT_FOLDER, f"{base_name}_result.jpg")

            try:
                # === 核心测量流程 === #
                measurer = get_measurer()
                result = measurer.measure(upload_path, save_vis_path=vis_path)

                if not result['success']:
                    return make_response(jsonify({
                        "code": 400,
                        "message": result['msg'],
                        "data": result['data']
                    }), 400)

                # 转换为URL路径
                # upload_url = convert_to_url_path(upload_path, app_root)
                # vis_url = None
                # if result.get('vis_path') and os.path.exists(result['vis_path']):
                #     vis_url = convert_to_url_path(result['vis_path'], app_root)

                # # 保存历史记录
                # try:
                #     user_id = request.headers.get('token')
                #     if user_id:
                #         history = CornWidthHistory(
                #             user_id=user_id,
                #             upload_path=upload_url,
                #             result_path=vis_url,
                #             length=result['data']['length'],
                #             center_width=result['data']['center_width'],
                #             max_width_mid_third=result['data']['max_width_mid_third'],
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
                    "message": "测量成功",
                    "data": {
                        "length": result['data']['length'],
                        "center_width": result['data']['center_width'],
                        "max_width_mid_third": result['data']['max_width_mid_third'],
                        # "original_image": upload_url,
                        # "result_image": vis_url
                    }
                }), 200)

            except ValueError as ve:
                logger.error(f"测量失败: {str(ve)}")
                return make_response(jsonify({
                    "code": 400,
                    "message": str(ve)
                }), 400)

            except Exception as e:
                logger.error(f"图像测量过程中出错: {str(e)}", exc_info=True)
                return make_response(jsonify({
                    "code": 500,
                    "message": f"图像测量过程中出错: {str(e)}"
                }), 500)

        except Exception as e:
            logger.error(f"服务器内部错误: {str(e)}", exc_info=True)
            return make_response(jsonify({
                "code": 500,
                "message": f"服务器内部错误: {str(e)}",
                "data": None
            }), 500)


# 注册命名空间
api.add_namespace(corn_lw_ns)