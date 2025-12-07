#!/usr/bin/env bash
set -e

# 1. Go to the folder where this script lives (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ">>> Using Python at: $(which python)"
echo ">>> Installing Python dependencies from requirements.txt ..."

cd assignment-1

docker build -t aisdp-assignment:latest .
docker run -it --name aisdp-container aisdp-assignment:latest


echo ">>> Pipeline finished successfully."