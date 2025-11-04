import os
import logging

logger = logging.getLogger(__name__)


def convert_to_url_path(file_path, app_root):
    """
    将文件系统路径转换为URL路径

    Args:
        file_path: 完整文件路径，如 /app/static/images/xxx.jpg
        app_root: 应用根目录，如 /app

    Returns:
        URL路径，如 /static/images/xxx.jpg

    Example:
        >>> convert_to_url_path('/app/static/images/test.jpg', '/app')
        '/static/images/test.jpg'
    """
    try:
        # 获取相对于app_root的路径
        rel_path = os.path.relpath(file_path, app_root)
        # 转换路径分隔符为URL格式
        url_path = rel_path.replace(os.sep, '/')
        # 确保以/开头
        if not url_path.startswith('/'):
            url_path = '/' + url_path
        return url_path
    except Exception as e:
        logger.error(f"路径转换失败: {file_path}, 错误: {str(e)}")
        return None