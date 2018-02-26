#!/usr/bin/env bash
# Install custom python package if requirements.txt is present
if [ -e "/requirements.txt" ]; then
    $(which pip) install --user -r /requirements.txt
fi

airflow initdb

if [ -n "$AIRFLOW_CONNECTION_ID" ] && [ -n "$AIRFLOW_CONNECTION_URI" ]; then
airflow connections --conn_id $AIRFLOW_CONNECTION_ID --conn_uri $AIRFLOW_CONNECTION_URI -a
fi

exec airflow webserver &
exec airflow scheduler