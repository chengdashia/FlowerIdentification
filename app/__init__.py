import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from flask_restx import Api
from flask_cors import CORS
import config

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


def create_app():
    # 获取应用根目录的绝对路径
    app_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    static_folder = os.path.join(app_root, 'static')
    
    app = Flask(__name__, static_folder=static_folder, static_url_path='/static')
    app.json.ensure_ascii = False  # 解决中文乱码问题
    # 加载配置
    app.config.from_object(config.DevelopmentConfig)

    db.init_app(app)
    # # 确保在这里导入所有模型
    # from app.models.user import User
    # from app.models.history import IdentifyHistory
    # # 其他模型...
    #
    # with app.app_context():
    #     db.create_all()
    api.init_app(app)
    CORS(app)

    # Logging configuration
    configure_logging(app)

    # Import and register blueprints
    from .routes.user import user_ns
    from .routes.base64.chr_identify import chr_identify_ns
    from .routes.chr_identify_file import chr_identify_file_ns
    from .routes.identify_history import history_ns
    from .routes.base64.corn_identify import corn_identify_ns
    from .routes.corn_filament import corn_filament_ns
    from .routes.base64.filament_identify import filament_identify_ns
    from .routes.corn_filament_nature import corn_filament_nature_ns
    from .routes.base64.leaf_sheath_identify import leaf_sheath_identify_ns
    from .routes.corn_leaf_sheath import corn_leaf_sheath_ns
    from .routes.base64.ym_analyzer import ym_analyzer_ns
    from .routes.corn_shape_analyzer import corn_shape_analyzer_ns
    from .routes.corn_all_analyzer import corn_all_analyzer_ns
    from .routes.statistics import statistics_ns
    from .routes.specific.appearance_grade import corn_grade_ns
    from .routes.specific.length_width import corn_lw_ns
    from .routes.specific.rachis_color import corn_rachis_color_ns
    from .routes.specific.top_color import corn_top_color_ns
    from .routes.specific.ear_row import corn_ear_row_ns

    api.add_namespace(user_ns, path='/auth')
    api.add_namespace(chr_identify_ns, path='/chr_identify')
    api.add_namespace(chr_identify_file_ns, path='/chr_identify_file')
    api.add_namespace(history_ns, path='/history')
    api.add_namespace(corn_identify_ns, path='/corn_identify')
    api.add_namespace(corn_filament_ns, path='/corn_filament')
    api.add_namespace(filament_identify_ns, path='/filament_identify')
    api.add_namespace(corn_filament_nature_ns, path='/corn_filament_nature')
    api.add_namespace(leaf_sheath_identify_ns, path='/leaf_sheath_identify')
    api.add_namespace(corn_leaf_sheath_ns, path='/corn_leaf_sheath')
    api.add_namespace(ym_analyzer_ns, path='/ym_analyzer')
    api.add_namespace(corn_shape_analyzer_ns, path='/corn_last_analyzer')
    api.add_namespace(corn_all_analyzer_ns, path='/corn_all_analyzer')
    api.add_namespace(statistics_ns, path='/statistics')
    # 玉米詳細分析
    api.add_namespace(corn_grade_ns, path='/corn_grade')
    api.add_namespace(corn_lw_ns, path='/corn_lw')
    api.add_namespace(corn_rachis_color_ns, path='/corn_rachis_color')
    api.add_namespace(corn_top_color_ns, path='/corn_top_color')
    api.add_namespace(corn_ear_row_ns, path='/corn_ear_row')

    # 添加根路由
    @app.route('/')
    def index():
        return render_template('main.html')

    # 错误处理程序
    @app.errorhandler(404)
    def not_found_error(error):
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

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    )

    # 创建一个旋转文件处理器
    file_handler = RotatingFileHandler('logs/myapp.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # 创建一个控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 配置根日志记录器，这样所有使用 logging.getLogger() 的记录器都会继承这个配置
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加处理器到根日志记录器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 将处理器添加到 Flask 的日志记录器中
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)

    # 设置日志级别
    app.logger.setLevel(logging.INFO)
    app.logger.info('MyApp 启动')
