#!/usr/bin/env bash
#
# Run a single Athena query, block until it finishes, and fail loudly.
# On success, prints the QueryExecutionId to stdout.
#
# Usage:
#   ./scripts/run_athena.sh "SELECT 1"
#   echo "SELECT 1" | ./scripts/run_athena.sh
#
# Config via environment variables:
#   ATHENA_WORKGROUP   Athena workgroup            (default: primary)
#   ATHENA_DATABASE    default database/catalog    (default: ccoaindex)
#   ATHENA_OUTPUT      S3 query-results location   (required)
#   ATHENA_POLL_SECS   poll interval in seconds    (default: 5)
#
set -euo pipefail

WORKGROUP="${ATHENA_WORKGROUP:-primary}"
DATABASE="${ATHENA_DATABASE:-ccoaindex}"
OUTPUT="${ATHENA_OUTPUT:?set ATHENA_OUTPUT to your Athena results S3 location, e.g. s3://my-bucket/athena/}"
POLL_SECS="${ATHENA_POLL_SECS:-5}"

# SQL from first arg, or from stdin if no arg given.
if [[ $# -ge 1 ]]; then
  SQL="$1"
else
  SQL="$(cat)"
fi

if [[ -z "${SQL// }" ]]; then
  echo "run_athena.sh: no SQL provided" >&2
  exit 2
fi

qid=$(aws athena start-query-execution \
  --query-string "$SQL" \
  --work-group "$WORKGROUP" \
  --query-execution-context "Database=$DATABASE" \
  --result-configuration "OutputLocation=$OUTPUT" \
  --output text --query 'QueryExecutionId')

while true; do
  state=$(aws athena get-query-execution --query-execution-id "$qid" \
    --output text --query 'QueryExecution.Status.State')
  case "$state" in
    SUCCEEDED)
      break
      ;;
    FAILED|CANCELLED)
      reason=$(aws athena get-query-execution --query-execution-id "$qid" \
        --output text --query 'QueryExecution.Status.StateChangeReason')
      echo "run_athena.sh: query $qid $state: $reason" >&2
      exit 1
      ;;
  esac
  sleep "$POLL_SECS"
done

echo "$qid"
