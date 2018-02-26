#!/bin/bash

set -eo pipefail

# setup Superset if we haven't already
if [ ! -f $SUPERSET_HOME/.admin-${ADMIN_USERNAME}-setup-complete ]; then
  echo "Running first time setup for Superset"

  echo "Creating admin user ${ADMIN_USERNAME}"
  cat > $SUPERSET_HOME/admin.config <<EOF
${ADMIN_USERNAME}
${ADMIN_FIRST_NAME}
${ADMIN_LAST_NAME}
${ADMIN_EMAIL}
${ADMIN_PWD}
${ADMIN_PWD}

EOF

  /bin/sh -c '/usr/local/bin/fabmanager create-admin --app superset < $SUPERSET_HOME/admin.config'

  rm $SUPERSET_HOME/admin.config

  echo "Initializing database"
  superset db upgrade

  echo "Creating default roles and permissions"
  superset init

  touch $SUPERSET_HOME/.admin-${ADMIN_USERNAME}-setup-complete
else
  # always upgrade the database, running any pending migrations
  superset db upgrade
fi

echo "Starting up Superset"
superset runserver
