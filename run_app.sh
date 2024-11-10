#!/bin/bash

# Start HTTP server in the background
python3 -m http.server --bind localhost &

# Run app.py
python3 app.py &

# Wait for a moment to allow the server to start
sleep 2

# Open the URL in the default browser
open http://[::1]:8000/
