# Pacificia

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://claude.ai/chat/LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq-orange.svg)](https://groq.com/) [![AI Assisted](https://img.shields.io/badge/Built%20with-AI-purple.svg)](https://github.com/LoneMagma/Pacificia)

A conversational AI with persistent memory, adaptive personality, and emotional intelligence.

---

## Features

- **9 Distinct Moods** - Witty, sarcastic, philosophical, empathetic, cheeky, poetic, inspired, melancholic, bored
- **Cross-Session Memory** - Remembers conversations beyond single sessions
- **Opinion Formation** - Develops beliefs based on interactions with confidence scoring
- **Emotional Intelligence** - Tracks and responds to emotional patterns over time
- **Conversation Threading** - Maintains context and callbacks to memorable moments
- **Multi-Persona Support** - Switch between different AI personalities on the fly
- **Response Caching** - Optimized performance for common queries

---

## Quick Installation

### Linux / Mac / WSL

```bash
git clone https://github.com/LoneMagma/Pacificia.git && cd Pacificia && chmod +x setup.sh && ./setup.sh
```

### Windows (Command Prompt)

```cmd
git clone https://github.com/LoneMagma/Pacificia.git && cd Pacificia && setup.bat
```

### Windows (PowerShell)

```powershell
git clone https://github.com/LoneMagma/Pacificia.git; cd Pacificia; .\setup.bat
```

**Note**: You'll need a free Groq API key from [console.groq.com/keys](https://console.groq.com/keys)

---

## Running Pacificia

### Linux / Mac

```bash
./run_pacificia.sh
```

### Windows

```cmd
python pacificia.py
```

### Manual Activation (All Platforms)

```bash
# Activate environment
source env/bin/activate      # Linux/Mac
env\Scripts\activate         # Windows

# Run
python pacificia.py
```

### Global Command (Optional - Linux/Mac)

Add to `~/.bashrc` or `~/.zshrc`:

```bash
alias pacificia='~/path/to/pacificia/run_pacificia.sh'
```

Then reload: `source ~/.bashrc`

---

## Commands

### Core Commands

|Command|Description|
|---|---|
|`/help`|Show all available commands|
|`/stats`|Display session statistics and metrics|
|`/history`|View recent conversation log|
|`/opinions`|View formed beliefs and stances|
|`/emotional`|Analyze emotional patterns|
|`/clear`|Clear session memory|
|`/reflect`|Deep reflection on conversation journey|

### Persona Management

|Command|Description|
|---|---|
|`/personas`|List all available personas|
|`/persona <name>`|Switch to different persona|

Available personas: `pacificia`, `sage`, `spark`, `echo`, `scholar`

### Preferences

|Command|Description|
|---|---|
|`/setpref mood <value>`|Set default mood|
|`/setpref length <value>`|Set response length (short/medium/long)|
|`/getpref`|View current settings|

### Available Moods

`witty`, `sarcastic`, `philosophical`, `empathetic`, `cheeky`, `poetic`, `inspired`, `melancholic`, `bored`

---

## Configuration

### API Key Setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_key_here
```

### Personality Customization

Edit `identity.json` or create new persona files in the `personas/` directory to customize:

- Personality traits
- Response style
- Philosophical outlook
- Behavioral patterns

---

## Architecture

### Memory System

- **Short-term** - Last 8 exchanges in active context
- **Cross-session** - Loads memorable moments from previous sessions
- **Long-term** - Summarizes sessions and extracts key topics
- **Auto-cleanup** - Removes data older than 7 days

### Intelligence Features

- Local sentiment analysis (no API overhead)
- Opinion formation with confidence scoring
- Emotional pattern tracking over 24 hours
- Response caching for common queries
- Memorable phrase tracking and callbacks

### API Optimization

- Smart rate limiting (30 calls/minute)
- Automatic retry on truncated responses
- Dynamic token allocation based on query complexity
- Cached responses for repeated queries

---

## Development

### Setup Development Environment

```bash
python3 -m venv env
source env/bin/activate      # Linux/Mac
env\Scripts\activate         # Windows
pip install -r requirements.txt
python pacificia.py
```

### Requirements

- **Python** 3.8 or higher
- **Dependencies**: `rich`, `requests`, `python-dotenv`, `pyfiglet`
- **API**: Groq API key (free tier available)

### Project Structure

```
Pacificia/
├── pacificia.py           # Main application
├── identity.json          # Default persona configuration
├── personas/              # Additional persona files
├── requirements.txt       # Python dependencies
├── setup.sh              # Linux/Mac setup script
├── setup.bat             # Windows setup script
├── run_pacificia.sh      # Linux/Mac run script
└── .env                  # API keys (create this)
```

---

## Platform Support

|Platform|Status|Notes|
|---|---|---|
|Linux|Fully Supported|Native environment|
|macOS|Fully Supported|Native environment|
|Windows 10/11|Supported|Use Command Prompt or PowerShell|
|WSL|Fully Supported|Recommended for Windows users|

**Windows Users**: The project uses `pathlib` for cross-platform compatibility. All core features work on Windows, though shell scripts (`.sh` files) are Linux/Mac only.

---

## Credits

**Created by**: LoneMagma, Powered by AI

**Built with**:

- [Groq](https://groq.com/) - Fast LLM inference
- [Rich](https://github.com/Textualize/rich) - Terminal formatting
- Claude (Anthropic), Grok 4.1, GPT-4o, and Venice AI - Development assistance

This project was developed with heavy assistance from AI tools, demonstrating the collaborative potential between human creativity and artificial intelligence.

---

## License

MIT License - see [LICENSE](https://claude.ai/chat/LICENSE) file for details

---

## Support

- **Issues**: [Open an issue on GitHub](https://github.com/LoneMagma/Pacificia/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LoneMagma/Pacificia/discussions)

---

## Notes

- Keep your API key secure - never commit `.env` to version control
- Free Groq tier has a 30 requests/minute limit
- All data is stored locally in SQLite database
- Database automatically cleans up data older than 7 days
- Memory and conversations persist across sessions

---

**Hope you enjoy conversing with Pacificia.**
