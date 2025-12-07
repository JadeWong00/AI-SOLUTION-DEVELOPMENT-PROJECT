#!/usr/bin/env bash
set -e

# 1. Go to the folder where this script lives (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

docker build -t aisdp-assignment:latest .
docker run --rm --name aisdp-container aisdp-assignment:latest