#!/bin/bash
# ─────────────────────────────────────────────
#  Pacificia Launcher
#  First run: sets up everything automatically.
#  After that: just launches.
# ─────────────────────────────────────────────

cd "$(dirname "$0")"

# Check Python
if ! command -v python3 &>/dev/null; then
    echo ""
    echo " Python 3 not found."
    echo " Install it first:"
    echo "   Ubuntu/Debian:  sudo apt install python3 python3-venv"
    echo "   macOS:          brew install python3"
    echo ""
    exit 1
fi

# First run: no virtual environment yet
if [ ! -f "env/bin/activate" ]; then
    echo " First run — setting up Pacificia..."
    echo ""
    chmod +x setup.sh
    ./setup.sh
    exit 0
fi

# Check API key
if [ -f ".env" ]; then
    if grep -q "your_groq_api_key_here" .env 2>/dev/null; then
        echo ""
        echo " API key not configured."
        echo " Edit .env and add your Groq key, then run this again."
        echo " Get a free key at: https://console.groq.com/keys"
        echo ""
        # Try to open editor
        if command -v nano &>/dev/null; then
            read -p " Open .env in nano now? (y/n): " ans
            [[ $ans =~ ^[Yy]$ ]] && nano .env
        fi
        exit 1
    fi
else
    echo " No .env file found. Creating from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
    else
        echo "GROQ_API_KEY=your_groq_api_key_here" > .env
    fi
    echo " Edit .env and add your Groq key, then run this again."
    command -v nano &>/dev/null && nano .env
    exit 1
fi

# Launch
source env/bin/activate
python3 pacificia.py
