FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /app

RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y python3-pip \
 && rm -rf /var/lib/apt/lists/* \
 && pip3 install pipenv

COPY ./Pipfile .
COPY ./Pipfile.lock .

COPY ./src ./src
COPY ./setup.py .
COPY ./README.md .
COPY ./LICENCE.md .

RUN pipenv install --system --deploy
