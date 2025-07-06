#!/bin/bash

apt-get update && apt-get install -y ffmpeg  # ✅ This line is essential

pip install --upgrade pip
pip install -r requirements.txt
