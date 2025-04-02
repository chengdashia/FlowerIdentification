from app import db


class IdentifyHistory(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    img = db.Column(db.Text, nullable=False, comment='花卉图片')
    user_id = db.Column(db.Integer, nullable=False, comment='用户id')
    prediction1 = db.Column(db.String(20), nullable=False, comment='预测结果1')
    probability1 = db.Column(db.Float, nullable=False, comment='可能性1')
    prediction2 = db.Column(db.String(20), nullable=False, comment='预测结果2')
    probability2 = db.Column(db.Float, nullable=False, comment='可能性2')
    created_time = db.Column(db.DateTime, default=db.func.current_timestamp())

    # 外键关联（如果需要）
    # user = db.relationship('User', backref=db.backref('identify_histories', lazy=True))

    def __repr__(self):
        return f'<GetIdentifyHistory {self.id}>'
