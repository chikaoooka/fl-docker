FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY src /app/src
WORKDIR /app

# エントリーポイントは設定しない。docker-composeで個別に指定する。