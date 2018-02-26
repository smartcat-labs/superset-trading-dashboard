FROM debian:stretch

# Superset version
ARG SUPERSET_VERSION=0.23.0rc4

# Configure environment
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONPATH=/etc/superset:/home/superset:$PYTHONPATH \
    SUPERSET_HOME=/var/lib/superset

# Create superset user & install dependencies
RUN useradd -U -m superset && \
    mkdir /etc/superset  && \
    mkdir ${SUPERSET_HOME} && \
    chown -R superset:superset /etc/superset && \
    chown -R superset:superset ${SUPERSET_HOME} && \
    apt-get update && \
    apt-get install -y \
        build-essential \
        curl \
        default-libmysqlclient-dev \
        libffi-dev \
        libldap2-dev \
        libpq-dev \
        libsasl2-dev \
        libssl-dev \
        openjdk-8-jdk \
        python3-dev \
        python3-pip && \
    pip3 install \
        flask-cors==3.0.3 \
        flask-mail==0.9.1 \
        flask-oauth==0.12 \
        flask_oauthlib==0.9.3 \
        gevent==1.2.2 \
        impyla==0.14.0 \
        mysqlclient==1.3.7 \
        psycopg2==2.6.1 \
        pyathenajdbc==1.2.0 \
        pyhive==0.5.0 \
        pyldap==2.4.28 \
        redis==2.10.5 \
        sqlalchemy-redshift==0.5.0 \
        sqlalchemy-clickhouse==0.1.1.post3 \
        Werkzeug==0.12.1

RUN apt-get install -y apt-transport-https wget
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
RUN echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list
RUN apt-get update && apt-get install -y yarn
RUN curl -sL https://deb.nodesource.com/setup_9.x | bash -
RUN apt-get update && apt-get install -y nodejs npm

RUN wget https://github.com/apache/incubator-superset/archive/0.23.0rc4.tar.gz
RUN tar -xvzf 0.23.0rc4.tar.gz
WORKDIR incubator-superset-0.23.0rc4/superset/assets
RUN yarn
RUN yarn run build
WORKDIR /incubator-superset-0.23.0rc4
RUN pip3 install setuptools
RUN python3 setup.py install


# Configure Filesystem
COPY script/superset-init.sh /superset-init.sh
RUN chmod +x /superset-init.sh
VOLUME /home/superset \
       /etc/superset \
       /var/lib/superset
WORKDIR /home/superset

# Deploy application
EXPOSE 8088
HEALTHCHECK CMD ["curl", "-f", "http://localhost:8088/health"]
ENTRYPOINT ["/superset-init.sh"]
USER superset