FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（包括 libglib2.0-0 和 OpenCV 常见依赖）
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk2.0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app

# 安装 Python 依赖
RUN pip install --upgrade pip \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
       --timeout=60 --retries=5 --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 5000

# 设置容器启动命令
CMD ["python", "run.py"]