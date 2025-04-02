class BaseConfig:
    SECRET_KEY = 'flower-identify'
    SQLALCHEMY_TRACK_MODIFICATIONS = False


# 开发环境
class DevelopmentConfig(BaseConfig):
    # 配置连接数据库
    HOSTNAME = '127.0.0.1'  # 服务器地址
    PORT = 3306  # 默认端口号
    USERNAME = 'root'
    PASSWORD = '12345678'
    DATABASE = 'flower_identify'  # 数据库名
    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4"


#  测试环境
class TestingConfig(BaseConfig):
    # 配置连接数据库
    HOSTNAME = '192.168.3.5'  # 服务器地址
    PORT = 3306  # 默认端口号
    USERNAME = 'root'
    PASSWORD = 'root'
    DATABASE = 'pythonbbs'  # 数据库名
    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4"


# 生产部署环境
class ProductionConfig(BaseConfig):
    # 配置连接数据库
    HOSTNAME = '134.175.18.239'  # 服务器地址
    PORT = 3306  # 默认端口号
    USERNAME = 'root'
    PASSWORD = 'root'
    DATABASE = 'pythonbbs'  # 数据库名
    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4"
