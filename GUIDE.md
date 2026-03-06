# Pacificia — Guide

A short reference for setup, running, and making Pacificia your own.

---

## Setup

**1. Get a free API key**
Go to [console.groq.com/keys](https://console.groq.com/keys) and create a key. It starts with `gsk_`.

**2. Add it to your `.env` file**
```
GROQ_API_KEY=gsk_your_key_here
```

**3. Choose how to run**

| Method | Command |
|--------|---------|
| Terminal (Linux/Mac) | `./setup.sh` then `pacificia` |
| Terminal (Windows) | `setup.bat` then `pacificia.bat` |
| Browser UI | Open `pacificia_ui.html` in any browser |

---

## Personas

Each persona has its own personality file. Switch mid-session with `/persona <name>`.

| Name | Style | File |
|------|-------|------|
| **Pacificia** | Witty, sardonic, sharp — the default | `identity_pacificia.json` |
| **Echo** | Empathetic, warm, listening-first | `identity_echo.json` |
| **Sage** | Philosophical, slow, wisdom-over-answers | `identity_sage.json` |
| **Scholar** | Analytical, structured, teaches the "why" | `identity_scholar.json` |
| **Spark** | Energetic, motivational, action-focused | `identity_spark.json` |

---

## Moods

Moods shift the tone within a persona. Set one with `/setpref mood <value>`.

`witty` · `sarcastic` · `philosophical` · `empathetic` · `cheeky` · `poetic` · `inspired` · `melancholic` · `bored`

---

## Commands

```
/help          Show all commands
/stats         Session statistics
/history       Recent conversation log
/opinions      Beliefs Pacificia has formed
/emotional     Emotional pattern analysis
/reflect       Deep reflection on the conversation
/clear         Clear session memory
/personas      List all personas
/persona name  Switch persona
/setpref mood <value>    Set default mood
/setpref length <value>  short / medium / long
/getpref       View your current settings
```

---

## Customizing a Persona

Open any `identity_*.json` and edit:

```json
{
  "name": "Pacificia",
  "core_style": {
    "tone": "change this",
    "voice": "and this"
  },
  "key_traits": [
    "Add or remove behavioral rules here"
  ]
}
```

**Rules of thumb:**
- `key_traits` are the most direct way to change behavior — they're read as instructions
- `philosophy` shapes how the persona thinks, not just how it speaks
- `response_guidelines.length` controls verbosity: `"1-2 sentences"` vs `"as long as needed"`
- `context_limit` is how many messages stay in active memory — lower = faster, higher = more coherent

---

## Creating a New Persona

Copy any existing identity file, rename it `identity_yourname.json`, and edit the fields.
Then add the name to the personas list in `pacificia.py` where personas are loaded.

---

## Memory

- **Short-term**: last few exchanges in the active context window
- **Long-term**: session summaries stored in a local SQLite `.db` file — never leaves your machine
- **Cleanup**: by default, data older than 7 days is pruned

To change the cleanup window, search for `days=7` in `pacificia.py` and adjust.

---

## Model

To switch the Groq model, add this to your `.env`:

```
GROQ_MODEL=llama3-70b-8192
```

Available models (as of writing): `llama3-8b-8192`, `llama3-70b-8192`, `mixtral-8x7b-32768`
Check [console.groq.com](https://console.groq.com) for the current list.

---

## The `.env` file — full reference

```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama3-70b-8192        # optional, defaults to llama3-70b
```

---

*Pacificia by LoneMagma — MIT License*
