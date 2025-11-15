#!/bin/bash
# start.sh — arranque para Render
# Asegúrate de tener permisos: chmod +x start.sh
uvicorn src.api:app --host 0.0.0.0 --port $PORT
