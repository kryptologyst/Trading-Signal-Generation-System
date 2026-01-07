#!/bin/bash
# Installation script for Trading Signal Generation System

echo "Installing Trading Signal Generation System..."
echo "=============================================="

# Check if Python 3.10+ is available
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✓ Python $python_version detected"
else
    echo "❌ Python 3.10+ required, found $python_version"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/cache
mkdir -p outputs
mkdir -p assets
mkdir -p logs

# Make scripts executable
chmod +x test_system.py
chmod +x main.py

echo ""
echo "Installation completed!"
echo ""
echo "To run the system:"
echo "  python3 main.py"
echo ""
echo "To run tests:"
echo "  python3 test_system.py"
echo ""
echo "To launch the demo:"
echo "  streamlit run demo/app.py"
echo ""
echo "DISCLAIMER: This software is for educational and research purposes only."
echo "It is not intended as investment advice and should not be used for actual"
echo "trading without proper risk management and professional consultation."
