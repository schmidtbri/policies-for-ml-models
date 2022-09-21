# syntax=docker/dockerfile:1
FROM python:3.9-slim

ARG BUILD_DATE

LABEL org.opencontainers.image.title="Policies for ML Models"
LABEL org.opencontainers.image.description="Policies for machine learning models."
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.authors="6666331+schmidtbri@users.noreply.github.com"
LABEL org.opencontainers.image.source="https://github.com/schmidtbri/policies-for-ml-models"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.licenses="MIT License"
LABEL org.opencontainers.image.base.name="python:3.9-slim"

WORKDIR /service

ARG USERNAME=service-user
ARG USER_UID=10000
ARG USER_GID=10000

# install packages
RUN apt-get update \
    && apt-get install --assume-yes --no-install-recommends sudo \
    && apt-get install --assume-yes --no-install-recommends git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# create a user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# installing dependencies
COPY ./service_requirements.txt ./service_requirements.txt
RUN pip install -r service_requirements.txt

# copying code, configuration, and license
COPY ./configuration ./configuration
COPY ./policy_decorator ./policy_decorator
COPY ./LICENSE ./LICENSE

USER $USERNAME

CMD ["uvicorn", "rest_model_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
