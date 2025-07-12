from flask import Blueprint, request, jsonify, make_response
from flask_restx import Resource, Namespace, fields
from app import api, db
from app.models.user import User
from app.models.history import ChrHistory, CornFilamentHistory, CornFilamentNatureHistory, CornLeafSheathHistory, CornShapeHistory
import logging
import os

logger = logging.getLogger(__name__)


history_bp = Blueprint('history', __name__)
history_ns = Namespace('history', description='Identify History related operations')

HISTORY_MODELS = {
    'chr': ChrHistory,
    'corn-filament-nature': CornFilamentNatureHistory,
    'corn-filament': CornFilamentHistory,
    'corn_leaf_sheath': CornLeafSheathHistory,
    'corn-shape': CornShapeHistory,
}


def get_absolute_path(file_path):
    """將相對路徑轉換為絕對路徑"""
    if not file_path:
        return None
    
    # 獲取應用根目錄
    app_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # 如果路徑以 /static 開頭，轉換為絕對路徑
    if file_path.startswith('/static/'):
        # 移除 /static/ 前綴，然後與應用根目錄拼接
        relative_path = file_path[8:]  # 移除 '/static/'
        return os.path.join(app_root, 'static', relative_path)
    
    # 如果路徑以 static 開頭，轉換為絕對路徑
    if file_path.startswith('static/'):
        return os.path.join(app_root, file_path)
    
    # 如果已經是絕對路徑，直接返回
    if os.path.isabs(file_path):
        return file_path
    
    # 其他情況，假設是相對於靜態目錄的路徑
    return os.path.join(app_root, 'static', file_path)


def delete_file_safely(file_path):
    """安全刪除文件，如果文件不存在則忽略錯誤，並刪除空的父目錄"""
    if not file_path:
        logger.info("文件路徑為空，跳過刪除")
        return True
        
    # 轉換為絕對路徑
    absolute_path = get_absolute_path(file_path)
    logger.info(f"原始路徑: {file_path}")
    logger.info(f"轉換後絕對路徑: {absolute_path}")
    
    # 檢查目錄是否存在
    if absolute_path:
        directory = os.path.dirname(absolute_path)
        logger.info(f"目錄路徑: {directory}")
        logger.info(f"目錄是否存在: {os.path.exists(directory)}")
        
        if os.path.exists(directory):
            try:
                files_in_dir = os.listdir(directory)
                logger.info(f"目錄中的文件: {files_in_dir}")
            except Exception as e:
                logger.error(f"無法列出目錄內容: {str(e)}")
    
    if absolute_path and os.path.exists(absolute_path):
        try:
            os.remove(absolute_path)
            logger.info(f"成功刪除文件: {absolute_path}")
            
            # 刪除文件後，檢查父目錄是否為空，如果是則刪除
            parent_dir = os.path.dirname(absolute_path)
            if parent_dir and os.path.exists(parent_dir):
                try:
                    # 檢查目錄是否為空
                    if not os.listdir(parent_dir):
                        # 檢查是否為results或uploads目錄，如果是則不刪除
                        dir_name = os.path.basename(parent_dir)
                        if dir_name not in ['results', 'uploads']:
                            os.rmdir(parent_dir)
                            logger.info(f"成功刪除空目錄: {parent_dir}")
                        else:
                            logger.info(f"跳過刪除系統目錄: {parent_dir}")
                except Exception as e:
                    logger.error(f"刪除空目錄失敗 {parent_dir}: {str(e)}")
            
            return True
        except Exception as e:
            logger.error(f"刪除文件失敗 {absolute_path}: {str(e)}")
            return False
    else:
        logger.warning(f"文件不存在: {absolute_path}")
        return True


def delete_history_files(record):
    """根據歷史記錄類型刪除相關的文件"""
    try:
        if isinstance(record, ChrHistory):
            # 刪除 ChrHistory 相關文件
            delete_file_safely(record.predicted_image_path)
            
        elif isinstance(record, CornFilamentHistory):
            # 刪除 CornFilamentHistory 相關文件
            delete_file_safely(record.upload_path)
            delete_file_safely(record.processed_image_path)
            
        elif isinstance(record, CornFilamentNatureHistory):
            # 刪除 CornFilamentNatureHistory 相關文件
            delete_file_safely(record.upload_path)
            delete_file_safely(record.white_background_path)
            delete_file_safely(record.red_region_path)
            delete_file_safely(record.comparison_path)
            
        elif isinstance(record, CornLeafSheathHistory):
            # 刪除 CornLeafSheathHistory 相關文件
            delete_file_safely(record.upload_path)
            delete_file_safely(record.white_background_path)
            delete_file_safely(record.red_region_path)
            delete_file_safely(record.comparison_path)
            
        elif isinstance(record, CornShapeHistory):
            # 刪除 CornShapeHistory 相關文件
            delete_file_safely(record.upload_path)
            delete_file_safely(record.cropped_ym_path)
            delete_file_safely(record.mask_path)
            delete_file_safely(record.overlay_path)
            delete_file_safely(record.analysis_path)
            
    except Exception as e:
        logger.error(f"Error deleting files for record {record.id}: {str(e)}")


@history_ns.route('/history-list', methods=['GET'])
class GetIdentifyHistory(Resource):
    @history_ns.doc(description='获取用户识别历史记录')
    @history_ns.param('page', '页码', type=int, default=1)
    @history_ns.param('pageSize', '每页数量', type=int, default=12)
    @history_ns.param('type', '记录类型', type=str, required=True, enum=list(HISTORY_MODELS.keys()))
    def get(self):
        user_id = request.headers.get('token')
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('pageSize', 12, type=int)
        history_type = request.args.get('type')

        if not user_id:
            return make_response(jsonify({"message": "User ID is required"}), 400)

        if not history_type or history_type not in HISTORY_MODELS:
            return make_response(jsonify({"message": "Invalid history type specified"}), 400)

        try:
            # 检查用户是否存在
            user = User.query.get(user_id)
            if user is None:
                return make_response(jsonify({"message": "User not found"}), 404)

            Model = HISTORY_MODELS[history_type]

            # 分页查询该用户的识别历史记录
            pagination = Model.query.filter_by(user_id=user_id) \
                .order_by(Model.created_time.desc()) \
                .paginate(page=page, per_page=page_size, error_out=False)

            records = pagination.items
            total = pagination.total

            records_data = [record.to_dict() for record in records]

            return make_response(jsonify({
                "code": 200,
                "message": "Success",
                "data": {
                    "records": records_data,
                    "total": total,
                    "page": page,
                    "pageSize": page_size
                }
            }), 200)

        except Exception as e:
            logger.error(f"Error fetching history list: {str(e)}")
            return make_response(jsonify({"message": f"Error: {str(e)}"}), 500)


@history_ns.route('/history-detail/<int:id>', methods=['GET'])
class GetHistoryDetail(Resource):
    @history_ns.doc(description='根据ID和类型获取用户识别历史记录详情')
    @history_ns.param('type', '记录类型', type=str, required=True, enum=list(HISTORY_MODELS.keys()))
    def get(self, id):
        user_id = request.headers.get('token')
        history_type = request.args.get('type')

        if not user_id:
            return make_response(jsonify({
                "code": 400,
                "message": "User ID is required"
            }), 400)

        if not history_type or history_type not in HISTORY_MODELS:
            return make_response(jsonify({"message": "Invalid history type specified"}), 400)

        try:
            # 检查用户是否存在
            user = User.query.get(user_id)
            if user is None:
                return make_response(jsonify({
                    "code": 400,
                    "message": "User not found"
                }), 400)

            Model = HISTORY_MODELS[history_type]

            # 查找记录
            record = Model.query.filter_by(id=id, user_id=user_id).first()
            if record is None:
                return make_response(jsonify({
                    "code": 404,
                    "message": "Record not found"
                }), 404)

            return make_response(jsonify({
                "code": 200,
                "message": "Get Record Detail successfully",
                "data": record.to_dict()
            }), 200)

        except Exception as e:
            logger.error(f"Error getting history detail: {str(e)}")
            return make_response(jsonify({
                "code": 500,
                "message": f"Error getting record: {str(e)}"
            }), 500)


@history_ns.route('/delete-history/<string:type>/<int:id>', methods=['DELETE'])
class DeleteIdentifyHistory(Resource):
    @history_ns.doc(description='根据ID和类型删除用户识别历史记录')
    @history_ns.param('type', '记录类型', type=str, required=True, enum=list(HISTORY_MODELS.keys()))
    def delete(self, type, id):
        user_id = request.headers.get('token')

        if not user_id:
            return make_response(jsonify({
                "code": 400,
                "message": "User ID is required"
            }), 400)

        if not type or type not in HISTORY_MODELS:
            return make_response(jsonify({"message": "Invalid history type specified"}), 400)

        try:
            # 检查用户是否存在
            user = User.query.get(user_id)
            if user is None:
                return make_response(jsonify({
                    "code": 400,
                    "message": "User not found"
                }), 400)

            Model = HISTORY_MODELS[type]

            # 查找记录
            record = Model.query.get(id)
            if record is None:
                return make_response(jsonify({
                    "code": 404,
                    "message": "Record not found"
                }), 404)

            # 校验记录是否属于当前用户
            if str(record.user_id) != str(user_id):
                return make_response(jsonify({
                    "code": 403,
                    "message": "Permission denied"
                }), 403)

            # 先刪除相關文件
            delete_history_files(record)

            # 删除记录
            db.session.delete(record)
            db.session.commit()

            return make_response(jsonify({
                "code": 200,
                "message": "Record deleted successfully"
            }), 200)

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error deleting history: {str(e)}")
            return make_response(jsonify({
                "code": 500,
                "message": f"Error deleting record: {str(e)}"
            }), 500)


@history_ns.route('/delete-history-bulk', methods=['POST'])
class BulkDeleteIdentifyHistory(Resource):
    @history_ns.doc(description='根据ID列表和类型批量删除用户识别历史记录')
    @history_ns.expect(api.model('BulkDeleteInput', {
        'ids': fields.List(fields.Integer, required=True, description='要删除的记录ID列表'),
        'type': fields.String(required=True, description='记录类型', enum=list(HISTORY_MODELS.keys()))
    }))
    def post(self):
        user_id = request.headers.get('token')
        data = request.get_json()
        ids = data.get('ids')
        history_type = data.get('type')

        if not user_id:
            return make_response(jsonify({"code": 400, "message": "User ID is required"}), 400)

        if not all([ids, history_type]) or history_type not in HISTORY_MODELS:
            return make_response(jsonify({"code": 400, "message": "Invalid input: `ids` and a valid `type` are required"}), 400)

        try:
            user = User.query.get(user_id)
            if user is None:
                return make_response(jsonify({"code": 400, "message": "User not found"}), 400)

            Model = HISTORY_MODELS[history_type]

            # 先獲取要刪除的記錄
            records_to_delete = Model.query.filter(Model.id.in_(ids), Model.user_id == user_id).all()
            
            if not records_to_delete:
                return make_response(jsonify({
                    "code": 404,
                    "message": "No matching records found to delete."
                }), 404)

            # 刪除相關文件
            for record in records_to_delete:
                delete_history_files(record)

            # 批量刪除數據庫記錄
            num_deleted = Model.query.filter(Model.id.in_(ids), Model.user_id == user_id).delete(synchronize_session=False)
            db.session.commit()

            return make_response(jsonify({
                "code": 200,
                "message": f"Successfully deleted {num_deleted} records."
            }), 200)

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error bulk deleting history: {str(e)}")
            return make_response(jsonify({"code": 500, "message": f"Error bulk deleting records: {str(e)}"}), 500)


# Add the namespace to the api
api.add_namespace(history_ns)
