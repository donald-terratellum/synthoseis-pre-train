import os
from pathlib import Path

# Root directory to search
ROOT = Path.home()

# Folders to skip
SKIP_FOLDER = [".venv", ".hermes", "Library", "node_modules", "venv", "env"]


def should_skip_dir(dirname: str) -> bool:
    return dirname == SKIP_FOLDER


def main():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # Remove .venv from traversal
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        # Check files in this directory
        for fname in filenames:
            if "session" in fname and "md" in fname:
                print(os.path.join(dirpath, fname))

if __name__ == "__main__":
    main()
