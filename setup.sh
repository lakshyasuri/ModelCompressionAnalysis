#!/bin/bash

echo "Setting up virtual environment..."

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install ipykernel

python -m ipykernel install --user --name=ModelCompressionAnalysis

echo "Setup complete. Run 'source .venv/bin/activate' to activate."