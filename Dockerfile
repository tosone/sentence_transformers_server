FROM ubuntu:22.04 as fetch

ARG USE_CHINA_MIRROR=false
RUN if [ "$USE_CHINA_MIRROR" = "true" ]; then \
        sed -i 's/ports.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list; \
    fi && \
    apt-get update && apt-get install -y python3-pip

RUN if [ "$USE_CHINA_MIRROR" = "true" ]; then \
        pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; \
    fi && \
    pip install "huggingface_hub[cli]" && \
    huggingface-cli download "BAAI/bge-m3" --repo-type model --local-dir /model --local-dir-use-symlinks False && ls -alh /model

FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

COPY --from=fetch /model /model

ARG USE_CHINA_MIRROR=false
RUN if [ "$USE_CHINA_MIRROR" = "true" ]; then \
        sed -i 's/ports.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list; \
    fi && \
    apt-get update && apt-get install -y python3 python3-pip

ENV ST_MODEL_NAME /model

WORKDIR /app

COPY . .

RUN if [ "$USE_CHINA_MIRROR" = "true" ]; then \
        pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; \
    fi && \
    python3 -m pip install -r requirements.txt

CMD ["python3", "server.py"]
