#!/usr/bin/env bash
# Install custom python package if requirements.txt is present
if [ -e "/requirements.txt" ]; then
    $(which pip) install --user -r /requirements.txt
fi

if [ -z "$FERNET_KEY" ]; then
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
fi

airflow initdb

if [ -n "$AIRFLOW_CONNECTION_ID" ] && [ -n "$AIRFLOW_CONNECTION_URI" ]; then
airflow connections --conn_id $AIRFLOW_CONNECTION_ID --conn_uri $AIRFLOW_CONNECTION_URI -a
fi

exec airflow webserver &
exec airflow scheduler