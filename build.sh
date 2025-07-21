#!/usr/bin/env bash
set -o errexit
apt-get update
apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-eng
pip install -r requirements.txt