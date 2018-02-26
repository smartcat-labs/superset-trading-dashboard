FROM python:3.6-slim-jessie

# Never prompts the user for choices on installation/configuration of packages
ENV DEBIAN_FRONTEND noninteractive
ENV TERM linux

# Airflow
ARG AIRFLOW_VERSION=1.9.0
ENV AIRFLOW_HOME=/usr/local/airflow
ENV AIRFLOW_LOGS=${AIRFLOW_LOGS}/logs
ENV AIRFLOW_LOGS_CONF_DIR=${AIRFLOW_HOME}/config
ENV DOCKER_VERSION=17.09.1~ce-0~debian

# Define en_US.
ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8
ENV LC_MESSAGES en_US.UTF-8
ENV LC_ALL en_US.UTF-8

RUN set -ex \
    && buildDeps=' \
        python3-dev \
        libkrb5-dev \
        libsasl2-dev \
        libssl-dev \
        libffi-dev \
        build-essential \
        libblas-dev \
        liblapack-dev \
        libpq-dev \
        git \
    ' \
    && apt-get update -yqq \
    && apt-get install -yqq --no-install-recommends \
        $buildDeps \
        python3-pip \
        python3-requests \
        apt-utils \
        curl \
        netcat \
        locales \
    && sed -i 's/^# en_US.UTF-8 UTF-8$/en_US.UTF-8 UTF-8/g' /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && useradd -ms /bin/bash -d ${AIRFLOW_HOME} airflow \
    && python -m pip install -U pip setuptools wheel \
    && pip install Cython \
    && pip install pytz \
    && pip install pyOpenSSL \
    && pip install ndg-httpsclient \
    && pip install pyasn1 \
    && pip install apache-airflow[crypto,celery,postgres,hive,jdbc]==$AIRFLOW_VERSION \
    && pip install celery[redis]==3.1.17 \
    && pip install paramiko \
    && apt-get purge --auto-remove -yqq $buildDeps \
    && apt-get clean \
    && rm -rf \
        /var/lib/apt/lists/* \
        /tmp/* \
        /var/tmp/* \
        /usr/share/man \
        /usr/share/doc \
        /usr/share/doc-base

# Install Docker
RUN echo "deb  http://deb.debian.org/debian  jessie main" >> /etc/apt/sources.list \
    && echo "deb-src  http://deb.debian.org/debian  jessie main" >> /etc/apt/sources.list \
    && apt-get update -y \
    && apt-get install \
       apt-transport-https \
       ca-certificates \
       curl \
       gnupg2 \
       software-properties-common -y \
    && curl -fsSL https://download.docker.com/linux/$(. /etc/os-release; echo "$ID")/gpg | apt-key add - \
    && add-apt-repository \
       "deb [arch=amd64] https://download.docker.com/linux/$(. /etc/os-release; echo "$ID") \
       $(lsb_release -cs) \
       stable" \
    && apt-get update \
    && apt-get install docker-ce=$DOCKER_VERSION -y

RUN mkdir -p ${AIRFLOW_LOGS_CONF_DIR}

RUN pip install docker-py

COPY script/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY config/airflow.cfg ${AIRFLOW_HOME}/airflow.cfg

EXPOSE 8090 5555 8793

ENV AIRFLOW_CONNECTION_ID=
ENV AIRFLOW_CONNECTION_URI=

WORKDIR ${AIRFLOW_HOME}
ENTRYPOINT ["/entrypoint.sh"]