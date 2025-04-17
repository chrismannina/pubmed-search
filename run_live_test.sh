#!/bin/bash

# Simple script to run the live_query_test.py with a specified query.

# --- Configuration ---
PYTHON_SCRIPT="live_query_test.py"
PYTHON_CMD="python" # Change if you use python3, etc.
API_BASE_URL="http://127.0.0.1:8000" # Must match the one used in the python script
API_CHECK_TIMEOUT=5 # Seconds to wait for API check

# --- Check for Query Argument ---
if [ -z "$1" ]; then
  echo "Usage: $0 \"<Your query string>\""
  echo "Example: $0 \"Latest treatments for migraine\""
  exit 1
fi

# --- Check if API is Running ---
echo "Checking if API server is running at $API_BASE_URL ..."
# Use GET instead of HEAD, as the root endpoint might only allow GET
if curl --output /dev/null --silent --fail --max-time $API_CHECK_TIMEOUT --request GET "$API_BASE_URL/"; then
  echo "API server is responding."
  echo "----------------------------------------"
else
  echo "Error: API server at $API_BASE_URL is not responding or failed the check."
  echo "Please ensure the FastAPI server (e.g., uvicorn api:app) is running."
  exit 1
fi

# --- Set Environment Variable ---
export LIVE_QUERY_STRING="$1"

# --- Run the Python Script ---
echo "Running live API test with query:"
echo "  $LIVE_QUERY_STRING"
echo "----------------------------------------"

# Ensure the python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found in the current directory."
    exit 1
fi

# Execute the script
$PYTHON_CMD "$PYTHON_SCRIPT"
EXIT_CODE=$?

echo "----------------------------------------"

# --- Report Status ---
if [ $EXIT_CODE -eq 0 ]; then
  echo "Python script finished successfully."
else
  echo "Python script finished with errors (Exit Code: $EXIT_CODE)."
fi

exit $EXIT_CODE 