FROM tensorflow/tensorflow:latest-gpu-py3 as base
WORKDIR /app

# idea - just don't rebuild if dependencies are not changed
# this will reinstall dependencies with every change to source, but will quite
# happily run newest source if given by volume
FROM base as dev
COPY ./setup.py .
COPY ./README.md .
COPY ./LICENCE.md .
COPY ./src ./src

RUN pip3 install -e .
