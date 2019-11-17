FROM tensorflow/tensorflow:latest-gpu-py3 as base
WORKDIR /app

FROM base as dirty
COPY ./src ./src
COPY ./setup.py .
COPY ./README.md .
COPY ./LICENCE.md .

RUN pip3 install .


# idea - just don't rebuild if dependencies are not changed
FROM dirty as dev
ENV PYTHONPATH=$PYTHONPATH:/app/src/
