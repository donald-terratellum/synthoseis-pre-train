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
