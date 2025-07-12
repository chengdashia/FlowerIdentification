from flask import Blueprint, jsonify, make_response
from flask_restx import Resource, Namespace
from datetime import datetime, timedelta
from app import db, api
from app.models.history import ChrHistory, CornFilamentHistory, CornFilamentNatureHistory, CornLeafSheathHistory, CornShapeHistory

statistics_bp = Blueprint('statistics', __name__)
statistics_ns = Namespace('statistics', description='统计与可视化相关接口')

HISTORY_MODELS = {
    'chr': ChrHistory,
    'filament': CornFilamentNatureHistory,
    'corn': CornFilamentHistory,
    'leaf_sheath': CornLeafSheathHistory,
    'ym': CornShapeHistory,
}

# 1. 一周内每日识别总次数
@statistics_ns.route('/week/total-count', methods=['GET'])
class WeekTotalCount(Resource):
    def get(self):
        today = datetime.now().date()
        week_days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
        result = []
        for day in week_days:
            count = 0
            for Model in HISTORY_MODELS.values():
                count += db.session.query(Model).filter(db.func.date(Model.created_time) == day).count()
            result.append({"date": day.strftime('%Y-%m-%d'), "count": count})
        return make_response(jsonify({"code": 200, "data": result}), 200)

# 2. 一周内各类别识别次数
@statistics_ns.route('/week/type-count', methods=['GET'])
class WeekTypeCount(Resource):
    def get(self):
        today = datetime.now().date()
        week_ago = today - timedelta(days=6)
        type_counts = {}
        for key, Model in HISTORY_MODELS.items():
            count = db.session.query(Model).filter(Model.created_time >= week_ago).count()
            type_counts[key] = count
        return make_response(jsonify({"code": 200, "data": type_counts}), 200)



# 注册到api（如用flask_restx）
api.add_namespace(statistics_ns)