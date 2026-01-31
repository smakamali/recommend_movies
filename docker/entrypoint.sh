#!/bin/sh
# Optional entrypoint: ensure dirs exist, then run CMD
mkdir -p /app/data /app/models /app/logs
exec "$@"
