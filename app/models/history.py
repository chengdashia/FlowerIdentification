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

    def to_dict(self):
        return {
            'id': self.id,
            'img': self.img,
            'user_id': self.user_id,
            'prediction1': self.prediction1,
            'probability1': self.probability1,
            'prediction2': self.prediction2,
            'probability2': self.probability2,
            'created_time': self.created_time.strftime('%Y-%m-%d %H:%M:%S') if self.created_time else None
        }

    def __repr__(self):
        return f'<GetIdentifyHistory {self.id}>'
