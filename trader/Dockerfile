FROM python:2

COPY requirements.txt ./trader/
RUN pip install --no-cache-dir -r trader/requirements.txt

COPY *.py ./trader/
COPY utils/*.py ./trader/utils/
RUN mkdir -p ./sqlite
RUN mkdir -p ./trader/data

ENV PYTHONPATH $PYTHONPATH:/trader/:/trader/utils/

WORKDIR ./trader