#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
export $(cat .env | grep -v ^# | xargs)
python main.py