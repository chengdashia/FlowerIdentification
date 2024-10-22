import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from flask_restx import Api
from flask_cors import CORS
import config
import torch
from torchvision import transforms
import pandas as pd
from app.models.network_structure import resnet34

# 初始化 SQLAlchemy
db = SQLAlchemy()
# 初始化 Flask-Restx 用于 API 文档
api = Api(
    version='1.0',
    title='My API',
    description='A simple API',
    terms_url='/terms',
    contact='your-email@example.com',
    license='MIT'
)


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_model(model_path):
    model = resnet34(include_top=True)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to('cpu')
    model.eval()
    return model


def create_app():
    app = Flask(__name__)
    app.config.from_object(config.DevelopmentConfig)

    db.init_app(app)
    api.init_app(app)
    CORS(app)

    # Load the model, transform, and label data and store them in the app config
    app.config['MODEL'] = load_model('static/best_model.pth')
    app.config['TRANSFORM'] = get_transform()
    app.config['LABEL_DATA'] = pd.read_csv('static/label.csv')

    # Logging configuration
    configure_logging(app)

    # Import and register blueprints
    from .routes.auth import auth_ns
    from .routes.identify import identify_ns
    from .routes.flower_identify import flower_identify_ns

    api.add_namespace(auth_ns, path='/auth')
    api.add_namespace(identify_ns, path='/identify')
    api.add_namespace(identify_ns, path='/flower_identify')

    # 错误处理程序
    @app.errorhandler(404)
    def not_found_error():
        app.logger.warning(f'Page not found: {request.url}')
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        app.logger.error('Server Error: %s', error)
        return render_template('500.html'), 500

    return app


# 配置日志记录函数
def configure_logging(app):
    # 移除默认的 Flask 日志处理器以防止重复日志
    del app.logger.handlers[:]

    # 创建日志目录
    if not os.path.exists('logs'):
        os.mkdir('logs')

    # 创建一个旋转文件处理器
    file_handler = RotatingFileHandler('logs/myapp.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)

    # 将处理器添加到 Flask 的日志记录器中
    app.logger.addHandler(file_handler)

    # 设置日志级别
    app.logger.setLevel(logging.INFO)
    app.logger.info('MyApp 启动')
