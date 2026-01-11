#!/usr/bin/env python3
"""
Simple HTTP server for viewing the lecture slides locally.
Run this script and visit http://localhost:8000 in your browser.
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

# Configure server
PORT = 8000
Handler = http.server.SimpleHTTPRequestHandler

# Get the current directory
current_dir = Path(__file__).parent.absolute()
os.chdir(current_dir)

print(f"Starting server at http://localhost:{PORT}")
print(f"Serving content from: {current_dir}")
print("Press Ctrl+C to stop the server")

# Open the browser automatically
webbrowser.open(f"http://localhost:{PORT}")

# Start the server
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.") 