#!/bin/sh

# Check if file already contains JSON (i.e., Kubernetes has decoded it)
if ! jq empty "$GOOGLE_APPLICATION_CREDENTIALS" >/dev/null 2>&1; then
  # If the file does not contain JSON, assume it's base64 and decode it
  echo $GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64 | base64 --decode > $GOOGLE_APPLICATION_CREDENTIALS
fi

# Execute command
exec python pipeline_dev.py
