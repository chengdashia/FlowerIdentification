from flask import Blueprint, request, jsonify, make_response
from flask_restx import Resource, Namespace
from app import api, db
from app.models.user import User
from app.models.history import IdentifyHistory
import logging

logger = logging.getLogger(__name__)


history_bp = Blueprint('history', __name__)
history_ns = Namespace('history', description='Identify History related operations')


@history_ns.route('/history-list', methods=['GET'])
class GetIdentifyHistory(Resource):
    @history_ns.doc(description='获取用户识别历史记录')
    @history_ns.param('page', '页码', type=int, default=1)
    @history_ns.param('pageSize', '每页数量', type=int, default=12)
    def get(self):
        user_id = request.headers.get('token')
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('pageSize', 12, type=int)

        if not user_id:
            return make_response(jsonify({"message": "User ID is required"}), 400)

        try:
            # 检查用户是否存在
            user = User.query.get(user_id)
            if user is None:
                return make_response(jsonify({"message": "User not found"}), 404)

            # 分页查询该用户的识别历史记录
            pagination = IdentifyHistory.query.filter_by(user_id=user_id) \
                .order_by(IdentifyHistory.created_time.desc()) \
                .paginate(page=page, per_page=page_size, error_out=False)

            records = pagination.items
            total = pagination.total

            records_data = []
            for record in records:
                records_data.append({
                    "id": record.id,
                    "img": record.img,
                    "prediction1": record.prediction1,
                    "probability1": record.probability1,
                    "prediction2": record.prediction2,
                    "probability2": record.probability2,
                    "create_time": record.created_time.strftime("%Y-%m-%d %H:%M:%S")
                })

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
            logger.error(f"Error: {str(e)}")
            return make_response(jsonify({"message": f"Error: {str(e)}"}), 500)


@history_ns.route('/history-detail/<int:id>', methods=['post'])
class GetHistoryDetail(Resource):
    @history_ns.doc(description='根据ID获取用户识别历史记录详情')
    def post(self, id):
        user_id = request.headers.get('token')

        if not user_id:
            return make_response(jsonify({
                "code": 400,
                "message": "User ID is required"
            }), 400)

        try:
            # 检查用户是否存在
            user = User.query.get(user_id)
            if user is None:
                return make_response(jsonify({
                    "code": 400,
                    "message": "User not found"
                }), 400)

            # 查找记录
            record = IdentifyHistory.query.filter_by(id=id, user_id=user_id).first()
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
            db.session.rollback()
            logger.error(f"Error: {str(e)}")
            return make_response(jsonify({
                "code": 500,
                "message": f"Error getting record: {str(e)}"
            }), 500)


@history_ns.route('/delete-history/<int:id>', methods=['DELETE'])
class DeleteIdentifyHistory(Resource):
    @history_ns.doc(description='根据ID删除用户识别历史记录')
    def delete(self, id):
        user_id = request.headers.get('token')

        if not user_id:
            return make_response(jsonify({
                "code": 400,
                "message": "User ID is required"
            }), 400)

        try:
            # 检查用户是否存在
            user = User.query.get(user_id)
            if user is None:
                return make_response(jsonify({
                    "code": 400,
                    "message": "User not found"
                }), 400)

            # 查找记录
            record = IdentifyHistory.query.get(id)
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
            logger.error(f"Error: {str(e)}")
            return make_response(jsonify({
                "code": 500,
                "message": f"Error deleting record: {str(e)}"
            }), 500)


# Add the namespace to the api
api.add_namespace(history_ns)
