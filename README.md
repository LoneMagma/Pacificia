<div align="center">

![Pacificia](./banner.svg)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](./LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-3572A5?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq-F55036?style=flat-square)](https://groq.com/)
[![Browser Ready](https://img.shields.io/badge/browser-no%20install-4af09a?style=flat-square)](#browser--no-install)

</div>

---

A terminal-based AI companion with persistent memory, adaptive mood, and five distinct personas. Not a productivity tool — a presence.

---

## Two ways to run

### Browser — no install

Download [`pacificia_ui.html`](./pacificia_ui.html), open it in any browser, enter your Groq API key. That's it. No Python, no terminal, no setup.

Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

---

### Terminal

**Windows** — double-click `Pacificia.bat`. First run sets everything up automatically.

**Linux / Mac** — run once, then re-run any time:
```bash
chmod +x Pacificia.sh && ./Pacificia.sh
```

**Manual setup:**
```bash
git clone https://github.com/LoneMagma/Pacificia.git
cd Pacificia
cp .env.example .env        # then add your GROQ_API_KEY
chmod +x setup.sh && ./setup.sh
```

---

## Personas

Five distinct personalities. Switch mid-session with `/persona <n>`.

| | Persona | Tone | Best for |
|---|---|---|---|
| 🟢 | **pacificia** | Witty, sardonic, adaptive | Default — sharp and honest |
| 🔵 | **echo** | Warm, empathetic, validating | Processing feelings, being heard |
| 🟡 | **sage** | Philosophical, patient, metaphorical | Perspective, big questions |
| 🟠 | **scholar** | Analytical, precise, structured | Explanations, deep dives |
| ⚡ | **spark** | Energetic, motivating, enthusiastic | Getting unstuck, momentum |

Persona files live in `personas/`. Add your own by copying any `identity_*.json`.

---

## Moods

Moods apply to **Pacificia only** — the others have fixed tonal identities.

```
witty  ·  sarcastic  ·  philosophical  ·  empathetic  ·  cheeky
poetic  ·  inspired  ·  melancholic  ·  bored
```

Switch with `/mood <n>`.

---

## Commands

```
/help              show all commands
/persona <n>       switch active persona
/mood <n>          change Pacificia's mood (Pacificia only)
/stats             session statistics
/history           recent conversation log
/opinions          views formed this session
/emotional         emotional pattern summary
/reflect           session reflection and summary
/clear             wipe session history
```

---

## Features

- **Persistent memory** — local SQLite, survives restarts, pruned by size not time
- **Opinion formation** — builds views over time with confidence scoring
- **Emotional tracking** — detects and responds to emotional patterns
- **Context threading** — callbacks to earlier moments in conversation
- **6 models** — switch between Groq models in settings
- **Browser UI** — full standalone HTML, no server, no install

---

## Project structure

```
Pacificia/
├── pacificia.py           # Terminal application
├── pacificia_ui.html      # Browser UI — open and run
│
├── personas/
│   ├── identity_pacificia.json
│   ├── identity_echo.json
│   ├── identity_sage.json
│   ├── identity_scholar.json
│   └── identity_spark.json
│
├── Pacificia.bat          # Windows launcher
├── Pacificia.sh           # Linux/Mac launcher
├── setup.bat / setup.sh   # First-run setup
│
├── .env.example           # Environment template
├── requirements.txt       # Python deps
├── GUIDE.md               # Customization reference
└── LICENSE
```

---

## Requirements

- Python 3.8+ *(terminal only — browser UI needs nothing)*
- A free [Groq API key](https://console.groq.com/keys)
- `rich` `requests` `python-dotenv` `pyfiglet`

---

## Platform support

| Platform | Status |
|---|---|
| Linux | ✓ |
| macOS | ✓ |
| Windows 10/11 | ✓ |
| WSL | ✓ |
| Any browser | ✓ |

---

## Notes

- `.env` is gitignored — never commit your API key
- Free Groq tier: ~30 requests/minute
- All data stored locally — nothing leaves your machine
- Browser UI: sessions are in-memory only, settings persist via `localStorage`

---

## Credits

Created by **[LoneMagma](https://github.com/LoneMagma)**

Built with [Groq](https://groq.com/) · [Rich](https://github.com/Textualize/rich) · [pyfiglet](https://github.com/pwaller/pyfiglet)

Developed with Claude, GPT-4o, and Grok.

---

<div align="center">
<sub>MIT License — see <a href="./LICENSE">LICENSE</a></sub>
</div>
