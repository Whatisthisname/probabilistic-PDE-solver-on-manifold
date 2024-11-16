#!/bin/bash

# Start HTTP server in the background
python3 -m http.server --directory visualization --bind localhost 8080
