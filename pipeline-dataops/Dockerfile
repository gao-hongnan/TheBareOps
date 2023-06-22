ARG PYTHON_VERSION=3.9
ARG CONTEXT_DIR=pipeline-dataops

# Base image
FROM python:${PYTHON_VERSION}-slim-buster as builder

# Set work directory to /pipeline-dataops
WORKDIR /pipeline-dataops

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements, to cache them in docker layer
ARG REQUIREMENTS=requirements.txt
ARG REQUIREMENTS_DEV=requirements_dev.txt
COPY ./${CONTEXT_DIR}/${REQUIREMENTS} .
COPY ./${CONTEXT_DIR}/${REQUIREMENTS_DEV} .

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r ${REQUIREMENTS} && \
    python3 -m pip install --no-cache-dir -r ${REQUIREMENTS_DEV} && \
    pip install -U gaohn-common-utils && \
    pip install pydantic==2.0b3

# This is the real runner for my app
FROM python:${PYTHON_VERSION}-slim-buster as runner

ARG CONTEXT_DIR=pipeline-dataops

# Install cron
RUN apt-get update && apt-get install -y cron

# Copy from builder image
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /pipeline-dataops /pipeline-dataops

# Set work dir again to the pipeline_training subdirectory
# Set the working directory inside the Docker container
WORKDIR /pipeline-dataops

ENV PATH="/opt/venv/bin:$PATH"

# set git commit env
ARG GIT_COMMIT_HASH
ENV GIT_COMMIT_HASH=${GIT_COMMIT_HASH}

# Copy the rest of the application
# COPY .env .env
COPY ${CONTEXT_DIR}/conf /pipeline-dataops/conf
COPY ${CONTEXT_DIR}/gcp-storage-service-account.json /pipeline-dataops/gcp-storage-service-account.json
COPY ${CONTEXT_DIR}/pipeline_training /pipeline-dataops/pipeline_training

CMD ["python", "-m", "pipeline_training.pipeline"]
