#!/bin/bash

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

clear

echo -e "${CYAN}"
cat << "EOF"
 ____             _  __ _      _       
|  _ \ __ _  ____(_)/ _(_) ___(_) __ _ 
| |_) / _` |/ __ | | |_| |/ __| |/ _` |
|  __/ (_| | (__ | |  _| | (__| | (_| |
|_|   \__,_|\____|_|_| |_|\___|_|\__,_|
                                        
EOF
echo -e "${NC}"
echo -e "${MAGENTA}════════════════════════════════════════${NC}"
echo -e "${GREEN}   Installing Pacificia AI Assistant${NC}"
echo -e "${MAGENTA}════════════════════════════════════════${NC}\n"

# Check if running with sudo (not recommended)
if [ "$EUID" -eq 0 ]; then 
    echo -e "${RED}⚠️  WARNING: Don't run this script with sudo!${NC}"
    echo "Run as normal user: ./setup.sh"
    exit 1
fi

# Step 1: Check Python
echo -e "${CYAN}[1/7]${NC} Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed.${NC}"
    echo ""
    echo "Please install Python 3.8 or higher:"
    echo -e "  ${BLUE}Ubuntu/Debian:${NC} sudo apt install python3 python3-pip python3-venv"
    echo -e "  ${BLUE}Fedora:${NC} sudo dnf install python3 python3-pip"
    echo -e "  ${BLUE}macOS:${NC} brew install python3"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')

echo -e "${GREEN}      ✅ Found Python $PYTHON_VERSION${NC}"

# Check if Python version is too old
if [ "$PYTHON_MINOR" -lt 8 ]; then
    echo -e "${RED}❌ Python 3.8 or higher is required (you have 3.$PYTHON_MINOR)${NC}"
    exit 1
fi

# Step 2: Check/Create virtual environment
echo -e "\n${CYAN}[2/7]${NC} Setting up virtual environment..."
if [ -d "env" ]; then
    echo -e "${GREEN}      ✅ Using existing virtual environment${NC}"
else
    python3 -m venv env
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}      ✅ Virtual environment created${NC}"
    else
        echo -e "${RED}      ❌ Failed to create virtual environment${NC}"
        exit 1
    fi
fi

# Step 3: Activate virtual environment
echo -e "\n${CYAN}[3/7]${NC} Activating virtual environment..."
source env/bin/activate
if [ $? -eq 0 ]; then
    echo -e "${GREEN}      ✅ Virtual environment activated${NC}"
else
    echo -e "${RED}      ❌ Failed to activate virtual environment${NC}"
    exit 1
fi

# Step 4: Upgrade pip
echo -e "\n${CYAN}[4/7]${NC} Upgrading pip..."
pip install --upgrade pip --quiet
echo -e "${GREEN}      ✅ Pip upgraded to $(pip --version | awk '{print $2}')${NC}"

# Step 5: Install dependencies
echo -e "\n${CYAN}[5/7]${NC} Installing dependencies..."
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}      ❌ requirements.txt not found!${NC}"
    exit 1
fi

echo "      Installing: rich, requests, python-dotenv, pyfiglet..."
pip install -r requirements.txt --quiet
if [ $? -eq 0 ]; then
    echo -e "${GREEN}      ✅ All dependencies installed${NC}"
else
    echo -e "${RED}      ❌ Failed to install dependencies${NC}"
    echo "      Try manually: pip install -r requirements.txt"
    exit 1
fi

# Step 6: Create .env file
echo -e "\n${CYAN}[6/7]${NC} Setting up environment variables..."
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}      ✅ Created .env from template${NC}"
    else
        cat > .env << 'ENVEOF'
# Pacificia Configuration
GROQ_API_KEY=your_groq_api_key_here
ENVEOF
        echo -e "${GREEN}      ✅ Created .env file${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}════════════════════════════════════════${NC}"
    echo -e "${YELLOW}   ⚠️  IMPORTANT: API Key Setup${NC}"
    echo -e "${YELLOW}════════════════════════════════════════${NC}"
    echo ""
    echo "Pacificia needs a Groq API key to work."
    echo ""
    echo -e "${BLUE}Option 1: Add it now (Recommended)${NC}"
    echo "  I'll open your .env file for editing"
    echo ""
    echo -e "${BLUE}Option 2: Add it later${NC}"
    echo "  You can edit .env manually anytime"
    echo ""
    
    read -p "Do you want to add your API key now? (y/n) [y]: " RESPONSE
    RESPONSE=${RESPONSE:-y}
    
    if [[ $RESPONSE =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${CYAN}📋 Quick Guide:${NC}"
        echo "  1. Get your FREE API key: ${BLUE}https://console.groq.com/keys${NC}"
        echo "  2. Copy the key (starts with 'gsk_...')"
        echo "  3. In the editor, replace 'your_groq_api_key_here' with your key"
        echo "  4. Save and close (Ctrl+O, Enter, Ctrl+X in nano)"
        echo ""
        read -p "Press Enter when ready to open the editor..."
        
        # Open editor
        if command -v nano &> /dev/null; then
            nano .env
        elif command -v vim &> /dev/null; then
            vim .env
        elif command -v vi &> /dev/null; then
            vi .env
        else
            echo -e "${YELLOW}      Please edit .env manually with your preferred editor${NC}"
        fi
        
        # Verify API key was added
        if grep -q "your_groq_api_key_here" .env; then
            echo ""
            echo -e "${YELLOW}⚠️  API key still not configured${NC}"
            echo "You'll need to add it before running Pacificia"
            echo -e "Edit with: ${GREEN}nano .env${NC}"
        else
            echo ""
            echo -e "${GREEN}✅ API key configured!${NC}"
        fi
    else
        echo ""
        echo -e "${YELLOW}⚠️  Remember to add your API key later:${NC}"
        echo -e "  ${GREEN}nano .env${NC}"
        echo ""
        echo "Get your key from: ${BLUE}https://console.groq.com/keys${NC}"
    fi
else
    echo -e "${GREEN}      ✅ .env file already exists${NC}"
    
    # Check if API key is set
    if grep -q "your_groq_api_key_here" .env 2>/dev/null; then
        echo -e "${YELLOW}      ⚠️  API key not configured yet!${NC}"
        echo "      Edit .env and add your Groq API key"
    fi
fi

# Step 7: Create global command
echo -e "\n${CYAN}[7/7]${NC} Creating global 'pacificia' command..."

INSTALL_DIR="$HOME/.local/bin"
PROJECT_DIR=$(pwd)
WRAPPER_SCRIPT="$INSTALL_DIR/pacificia"

# Create .local/bin if it doesn't exist
mkdir -p "$INSTALL_DIR"

# Create wrapper script
cat > "$WRAPPER_SCRIPT" << EOF
#!/bin/bash
# Pacificia launcher script
# Auto-generated by setup.sh

cd "$PROJECT_DIR" || exit 1
source env/bin/activate 2>/dev/null || {
    echo "Error: Virtual environment not found!"
    echo "Run setup.sh again from: $PROJECT_DIR"
    exit 1
}
python3 pacificia.py "\$@"
EOF

chmod +x "$WRAPPER_SCRIPT"
echo -e "${GREEN}      ✅ Command created at: $WRAPPER_SCRIPT${NC}"

# Check if already in PATH
if [[ ":$PATH:" == *":$INSTALL_DIR:"* ]]; then
    echo -e "${GREEN}      ✅ $INSTALL_DIR already in PATH${NC}"
    NEED_RELOAD=false
else
    # Detect shell
    SHELL_RC=""
    if [ -n "$BASH_VERSION" ]; then
        SHELL_RC="$HOME/.bashrc"
        SHELL_NAME="Bash"
    elif [ -n "$ZSH_VERSION" ]; then
        SHELL_RC="$HOME/.zshrc"
        SHELL_NAME="Zsh"
    else
        SHELL_RC="$HOME/.profile"
        SHELL_NAME="Shell"
    fi
    
    # Check if already in config file
    if grep -q "export PATH=\"\$HOME/.local/bin:\$PATH\"" "$SHELL_RC" 2>/dev/null; then
        echo -e "${GREEN}      ✅ Already in $SHELL_RC${NC}"
        NEED_RELOAD=true
    else
        # Add to config
        echo "" >> "$SHELL_RC"
        echo "# Added by Pacificia setup" >> "$SHELL_RC"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
        echo -e "${GREEN}      ✅ Added to $SHELL_RC${NC}"
        NEED_RELOAD=true
    fi
fi

# Installation complete
echo ""
echo -e "${MAGENTA}════════════════════════════════════════${NC}"
echo -e "${GREEN}   ✨ Installation Complete!${NC}"
echo -e "${MAGENTA}════════════════════════════════════════${NC}"
echo ""

# Check if API key is configured
if [ -f .env ]; then
    if grep -q "your_groq_api_key_here" .env; then
        echo -e "${RED}⚠️  API Key Not Configured!${NC}"
        echo ""
        echo "Before running Pacificia, add your API key:"
        echo -e "  ${GREEN}nano .env${NC}"
        echo ""
        echo "Get your FREE key: ${BLUE}https://console.groq.com/keys${NC}"
        echo ""
    else
        echo -e "${GREEN}✅ API key is configured${NC}"
        echo ""
    fi
fi

# Next steps
echo -e "${CYAN}Quick Start:${NC}"
echo ""

if [ "$NEED_RELOAD" = true ]; then
    echo -e "${YELLOW}1. Reload your shell:${NC}"
    if [ -n "$BASH_VERSION" ]; then
        echo -e "   ${GREEN}source ~/.bashrc${NC}"
    elif [ -n "$ZSH_VERSION" ]; then
        echo -e "   ${GREEN}source ~/.zshrc${NC}"
    else
        echo -e "   ${GREEN}source ~/.profile${NC}"
    fi
    echo ""
    echo -e "${YELLOW}2. Run Pacificia:${NC}"
else
    echo -e "${YELLOW}Run Pacificia:${NC}"
fi

echo -e "   ${GREEN}pacificia${NC}"
echo ""

echo -e "${CYAN}────────────────────────────────────────${NC}"
echo -e "${BLUE}Happy chatting!${NC}"
echo -e "${CYAN}────────────────────────────────────────${NC}"
echo ""

# Test if command works immediately
if command -v pacificia &> /dev/null; then
    echo -e "${GREEN}✨ 'pacificia' command is ready to use!${NC}"
    echo ""
else
    echo -e "${YELLOW}Note: Reload your shell first (see above)${NC}"
    echo ""
fi
