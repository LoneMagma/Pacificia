# Contributing to Pacificia

Thanks for your interest. Pacificia is a personal project — contributions are welcome but the scope is intentionally narrow.

## What fits

- Bug fixes in `pacificia.py`
- New persona files (`identity_*.json`)
- Improvements to setup scripts (`setup.sh`, `setup.bat`)
- UI fixes for `pacificia_ui.html`

## What doesn't fit

Pacificia is a terminal companion with persistent memory and multiple personas. It is not a general-purpose assistant, productivity tool, or platform. Feature requests that add capabilities outside that scope will be closed.

## How to contribute

1. Fork the repo
2. Create a branch: `git checkout -b fix/your-fix`
3. Make your change
4. Test it locally with a valid Groq API key
5. Open a pull request with a clear description

## Adding a persona

Copy any `identity_*.json`, rename it `identity_yourname.json`, and edit the fields. The filename must match the persona name (lowercase). Add the name to the persona loader in `pacificia.py`.

## Code style

- No new dependencies without discussion
- Keep `pacificia.py` as a single file
- Match the existing tone in identity files — read the others before writing one
