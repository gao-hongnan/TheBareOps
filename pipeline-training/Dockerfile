ARG PYTHON_VERSION=3.9
ARG CONTEXT_DIR=.
ARG HOME_DIR=/pipeline-training
ARG VENV_DIR=/opt
ARG VENV_NAME=venv

# Base image
FROM python:${PYTHON_VERSION}-slim-buster as builder

ARG CONTEXT_DIR
ARG HOME_DIR
ARG VENV_DIR
ARG VENV_NAME

# Set work directory to /pipeline-training
WORKDIR ${HOME_DIR}

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv ${VENV_DIR}/${VENV_NAME}
ENV PATH="${VENV_DIR}/${VENV_NAME}/bin:$PATH"

# Copy only requirements, to cache them in docker layer
ARG REQUIREMENTS=requirements.txt
ARG REQUIREMENTS_DEV=requirements_dev.txt
COPY ./${CONTEXT_DIR}/${REQUIREMENTS} .
COPY ./${CONTEXT_DIR}/${REQUIREMENTS_DEV} .

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r ${REQUIREMENTS} && \
    python3 -m pip install --no-cache-dir -r ${REQUIREMENTS_DEV} && \
    pip install -U gaohn-common-utils

# This is the real runner for my app
FROM python:${PYTHON_VERSION}-slim-buster as runner

ARG CONTEXT_DIR
ARG HOME_DIR
ARG VENV_DIR
ARG VENV_NAME

# Copy from builder image
COPY --from=builder ${VENV_DIR}/${VENV_NAME} ${VENV_DIR}/${VENV_NAME}
COPY --from=builder ${HOME_DIR} ${HOME_DIR}

# Set work dir again to the pipeline_training subdirectory
# Set the working directory inside the Docker container
WORKDIR ${HOME_DIR}

# Install jq
RUN apt-get update && \
    apt-get install -y --no-install-recommends jq && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="${VENV_DIR}/${VENV_NAME}/bin:$PATH"

# set git commit env
ARG GIT_COMMIT_HASH
ENV GIT_COMMIT_HASH=${GIT_COMMIT_HASH}

# Copy the rest of the application's code
COPY ${CONTEXT_DIR}/conf ${HOME_DIR}/conf
COPY ${CONTEXT_DIR}/metadata ${HOME_DIR}/metadata
COPY ${CONTEXT_DIR}/schema ${HOME_DIR}/schema
COPY ${CONTEXT_DIR}/pipeline_training ${HOME_DIR}/pipeline_training
COPY ${CONTEXT_DIR}/pipeline_dev.py ${HOME_DIR}/pipeline_dev.py
COPY ${CONTEXT_DIR}/scripts/docker/entrypoint.sh ${HOME_DIR}/scripts/docker/entrypoint.sh

RUN chmod -R +x ${HOME_DIR}/scripts/docker

CMD ["scripts/docker/entrypoint.sh"]