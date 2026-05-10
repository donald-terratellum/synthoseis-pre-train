# Copilot Instructions for synthoseis-pre-train

This repository uses a `uv` Python environment managed from the repository root.

## Environment setup
- Use `pyproject.toml` as the dependency manifest.
- Use `.python-version` to pin the Python interpreter to `3.11`.
- Install dependencies with:
  ```bash
  uv sync
  ```
- Run Python commands through the `uv` environment:
  ```bash
  uv run python train.py --data_path /path/to/seismic.zarr --epochs 100 --batch_size 4
  uv run python test_gpu_resources.py
  uv run python inference.py --sample_shape 128 128 128 --batch_size 1 --device auto
  ```

## Code organization
- Keep application entrypoints at the repository root:
  - `train.py`
  - `test_gpu_resources.py`
  - `inference.py`
- Keep reusable modules inside the `src/` package.
- Do not use `requirements.txt` for dependency management.
- If new dependencies are required, add them to `pyproject.toml` and then run `uv sync`.

## Best practices
- Prefer `uv run` over using system Python directly.
- Use `uv python install 3.11` if Python 3.11 is not already available.
- When testing, use `uv run python -m pytest`.
- When formatting or linting, use `uv run python -m black .`.

## Note for code modifications
- Ensure `src/` package imports remain correct.
- Avoid adding environment artifacts into version control; `.gitignore` already excludes `.venv/`, `uv.lock`, and Python cache files.

## Host And Path Mapping (Mac Mini + MacBook)

This repository lives on the **Mac mini** and is network-mounted on the **MacBook**.
The same physical files appear at two paths depending on which machine is used:

| Machine | Path prefix | Can edit files? | Can run code? |
|---|---|---|---|
| Mac mini (`Donalds-Mac-mini`) | `/Users/donaldpg/synthoseis-pre-train` | Yes | Yes |
| MacBook (`Donalds-MacBook-Pro`) | `/Volumes/donaldpg/synthoseis-pre-train` | Yes (via VS Code) | No |

**VS Code runs on the MacBook** and opens the repo via the network mount at
`/Volumes/donaldpg/synthoseis-pre-train`. Copilot's file-edit tools (read_file,
replace_string_in_file, create_file, etc.) operate on this path and changes are
immediately visible on the Mac mini at `/Users/donaldpg/synthoseis-pre-train` —
they are the same physical files.

**Code execution (terminals, pytest, train.py) happens on the Mac mini**, not the
MacBook. Copilot cannot run commands on the Mac mini. The user pastes terminal
output from the Mac mini into the chat.

### Rules for Copilot responses

**File editing:**
- Always use the `/Volumes/donaldpg/synthoseis-pre-train` path when calling file tools.
- Edits made via file tools are immediately visible at `/Users/donaldpg/...` on the Mac mini.
- Never suggest editing files by running shell commands — use file tools directly.

**Command suggestions:**
- When suggesting shell commands for the user to run, always use the Mac mini path:
  `/Users/donaldpg/synthoseis-pre-train`
- Treat the terminal hostname in pasted output as the source of truth for execution context.
- Do not assume `/Volumes/...` exists on the Mac mini.
- Do not assume `/Users/...` on the MacBook refers to the same repo unless confirmed.

**Preferred workflow:**
1. Copilot edits files directly using file tools (no need for the user to run git commands or copy-paste code).
2. User runs the result on the Mac mini and pastes output back into chat.
3. Copilot reads the output and iterates via more file edits.
