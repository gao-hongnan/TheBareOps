#!/bin/sh

# Decode the base64 encoded JSON credentials and write them to a file
echo $GOOGLE_APPLICATION_CREDENTIALS_JSON | base64 --decode > $GOOGLE_APPLICATION_CREDENTIALS

# Execute command
exec python pipeline.py
