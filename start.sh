#!/bin/bash

# Script to start the lecture series server
# By: Armin Mehrabian
# For: GWU ECE 6125: Parallel Computer Architecture

echo "Starting GWU ECE 6125 Parallel Computer Architecture Lecture Series Server..."

# Change to the script's directory
cd "$(dirname "$0")"

# Default port
PORT=8000

# Check if a port number is provided as an argument
if [ $# -eq 1 ]; then
    if [[ $1 =~ ^[0-9]+$ ]]; then
        PORT=$1
    else
        echo "Warning: Invalid port number. Using default port $PORT."
    fi
fi

echo "Server will be available at: http://localhost:$PORT"
echo "Press Ctrl+C to stop the server."
echo ""

# Start the Python web server
python3 -m http.server $PORT

# This line will only execute if the server is stopped
echo "Server stopped." 