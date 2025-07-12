from app import db


class CornFilamentNatureHistory(db.Model):
    __tablename__ = 'corn_filament_nature_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, comment='用户id')
    upload_path = db.Column(db.String(255), nullable=False, comment='上传图片路径')
    white_background_path = db.Column(db.String(255), comment='白底图片路径')
    red_region_path = db.Column(db.String(255), comment='红色区域图片路径')
    comparison_path = db.Column(db.String(255), comment='对比图路径')
    original_lab_l = db.Column(db.Float, comment='原始图像L值')
    original_lab_a = db.Column(db.Float, comment='原始图像A值')
    original_lab_b = db.Column(db.Float, comment='原始图像B值')
    red_region_lab_l = db.Column(db.Float, comment='红色区域L值')
    red_region_lab_a = db.Column(db.Float, comment='红色区域A值')
    red_region_lab_b = db.Column(db.Float, comment='红色区域B值')
    created_time = db.Column(db.DateTime, default=db.func.current_timestamp())

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'upload_path': self.upload_path,
            'white_background_path': self.white_background_path,
            'red_region_path': self.red_region_path,
            'comparison_path': self.comparison_path,
            'original_lab': {'L': self.original_lab_l, 'A': self.original_lab_a, 'B': self.original_lab_b},
            'red_region_lab': {'L': self.red_region_lab_l, 'A': self.red_region_lab_a, 'B': self.red_region_lab_b},
            'created_time': self.created_time.strftime('%Y-%m-%d %H:%M:%S') if self.created_time else None
        }


class CornFilamentHistory(db.Model):
    __tablename__ = 'corn_filament_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, comment='用户id')
    upload_path = db.Column(db.String(255), nullable=False, comment='上传图片路径')
    processed_image_path = db.Column(db.String(255), comment='处理后图片路径')
    mean_lab_l = db.Column(db.Float, comment='平均L值')
    mean_lab_a = db.Column(db.Float, comment='平均A值')
    mean_lab_b = db.Column(db.Float, comment='平均B值')
    max_a_lab_l = db.Column(db.Float, comment='最大A值对应L值')
    max_a_lab_a = db.Column(db.Float, comment='最大A值')
    max_a_lab_b = db.Column(db.Float, comment='最大A值对应B值')
    created_time = db.Column(db.DateTime, default=db.func.current_timestamp())

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'upload_path': self.upload_path,
            'processed_image_path': self.processed_image_path,
            'mean_lab': {'L': self.mean_lab_l, 'A': self.mean_lab_a, 'B': self.mean_lab_b},
            'max_a_lab': {'L': self.max_a_lab_l, 'A': self.max_a_lab_a, 'B': self.max_a_lab_b},
            'created_time': self.created_time.strftime('%Y-%m-%d %H:%M:%S') if self.created_time else None
        }


class ChrHistory(db.Model):
    __tablename__ = 'chr_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, comment='用户id')
    predicted_image_path = db.Column(db.String(255), nullable=False, comment='被预测的图片路径')
    prediction1 = db.Column(db.String(20), nullable=False, comment='预测结果1')
    probability1 = db.Column(db.Float, nullable=False, comment='可能性1')
    prediction2 = db.Column(db.String(20), nullable=False, comment='预测结果2')
    probability2 = db.Column(db.Float, nullable=False, comment='可能性2')
    created_time = db.Column(db.DateTime, default=db.func.current_timestamp())

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'predicted_image_path': self.predicted_image_path,
            'prediction1': self.prediction1,
            'probability1': self.probability1,
            'prediction2': self.prediction2,
            'probability2': self.probability2,
            'created_time': self.created_time.strftime('%Y-%m-%d %H:%M:%S') if self.created_time else None
        }


class CornLeafSheathHistory(db.Model):
    __tablename__ = 'corn_leaf_sheath_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, comment='用户id')
    upload_path = db.Column(db.String(255), nullable=False, comment='上传图片路径')
    white_background_path = db.Column(db.String(255), comment='白底图片路径')
    red_region_path = db.Column(db.String(255), comment='红色区域图片路径')
    comparison_path = db.Column(db.String(255), comment='对比图路径')
    original_lab_l = db.Column(db.Float, comment='原始图像L值')
    original_lab_a = db.Column(db.Float, comment='原始图像A值')
    original_lab_b = db.Column(db.Float, comment='原始图像B值')
    red_region_lab_l = db.Column(db.Float, comment='红色区域L值')
    red_region_lab_a = db.Column(db.Float, comment='红色区域A值')
    red_region_lab_b = db.Column(db.Float, comment='红色区域B值')
    created_time = db.Column(db.DateTime, default=db.func.current_timestamp())

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'upload_path': self.upload_path,
            'white_background_path': self.white_background_path,
            'red_region_path': self.red_region_path,
            'comparison_path': self.comparison_path,
            'original_lab': {'L': self.original_lab_l, 'A': self.original_lab_a, 'B': self.original_lab_b},
            'red_region_lab': {'L': self.red_region_lab_l, 'A': self.red_region_lab_a, 'B': self.red_region_lab_b},
            'created_time': self.created_time.strftime('%Y-%m-%d %H:%M:%S') if self.created_time else None
        }


class CornShapeHistory(db.Model):
    __tablename__ = 'corn_shape_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, comment='用户id')
    upload_path = db.Column(db.String(255), nullable=False, comment='上传图片路径')
    cropped_ym_path = db.Column(db.String(255), comment='裁剪后YM图片路径')
    mask_path = db.Column(db.String(255), comment='掩码图片路径')
    overlay_path = db.Column(db.String(255), comment='叠加图片路径')
    analysis_path = db.Column(db.String(255), comment='分析图片路径')
    shape_type = db.Column(db.String(50), comment='形状类型')
    width1 = db.Column(db.Float, comment='中位线1宽度')
    width2 = db.Column(db.Float, comment='中位线2宽度')
    width3 = db.Column(db.Float, comment='中位线3宽度')
    width4 = db.Column(db.Float, comment='中位线4宽度')
    width5 = db.Column(db.Float, comment='中位线5宽度')
    ratio_1_3 = db.Column(db.Float, comment='1/3比值')
    ratio_3_5 = db.Column(db.Float, comment='3/5比值')
    ratio_1_5 = db.Column(db.Float, comment='1/5比值')
    created_time = db.Column(db.DateTime, default=db.func.current_timestamp())

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'upload_path': self.upload_path,
            'cropped_ym_path': self.cropped_ym_path,
            'mask_path': self.mask_path,
            'overlay_path': self.overlay_path,
            'analysis_path': self.analysis_path,
            'shape_type': self.shape_type,
            'midline_widths': [self.width1, self.width2, self.width3, self.width4, self.width5],
            'ratios': {'ratio_1_3': self.ratio_1_3, 'ratio_3_5': self.ratio_3_5, 'ratio_1_5': self.ratio_1_5},
            'created_time': self.created_time.strftime('%Y-%m-%d %H:%M:%S') if self.created_time else None
        }
