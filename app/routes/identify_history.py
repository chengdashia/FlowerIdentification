from flask import Blueprint, request, jsonify, make_response
from flask_restx import Resource, Namespace, fields
from app import api, db
from app.models.user import User
from app.models.history import ChrHistory, CornHistory, FilamentHistory, LeafSheathHistory, YmHistory
import logging

logger = logging.getLogger(__name__)


history_bp = Blueprint('history', __name__)
history_ns = Namespace('history', description='Identify History related operations')

HISTORY_MODELS = {
    'chr': ChrHistory,
    'filament': FilamentHistory,
    'corn': CornHistory,
    'leaf_sheath': LeafSheathHistory,
    'ym': YmHistory,
}


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

            # 批量删除属于该用户的记录
            num_deleted = Model.query.filter(Model.id.in_(ids), Model.user_id == user_id).delete(synchronize_session=False)
            db.session.commit()

            if num_deleted > 0:
                return make_response(jsonify({
                    "code": 200,
                    "message": f"Successfully deleted {num_deleted} records."
                }), 200)
            else:
                return make_response(jsonify({
                    "code": 404,
                    "message": "No matching records found to delete."
                }), 404)

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error bulk deleting history: {str(e)}")
            return make_response(jsonify({"code": 500, "message": f"Error bulk deleting records: {str(e)}"}), 500)


# Add the namespace to the api
api.add_namespace(history_ns)
