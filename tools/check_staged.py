"""Check the actual staged blobs without printing their potentially private contents."""

import re
import subprocess
import sys
from pathlib import PurePosixPath

BLOCKED = {".mp4", ".mkv", ".png", ".jpg", ".jpeg", ".so", ".onnx", ".pt", ".safetensors", ".whl"}
PATTERNS = {
    "credential": re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    "private-home-path": re.compile(rb"/(?:home|root)/[A-Za-z0-9_.-]+/"),
    "private-network": re.compile(
        rb"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
    ),
}


def main():
    names = (
        subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
        )
        .decode()
        .split("\0")
    )
    errors = []
    for name in filter(None, names):
        path = PurePosixPath(name)
        if (
            (path.name.startswith(".env") and path.name != ".env.example")
            or path.suffix.lower() in BLOCKED
            or any(x in {"runs", "data", "build", "dist", ".venv"} for x in path.parts)
        ):
            errors.append((name, "artifact-or-env"))
            continue
        data = subprocess.check_output(["git", "show", ":" + name])
        if b"\0" in data or len(data) > 1024 * 1024:
            errors.append((name, "binary-or-large"))
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                errors.append((name, label))
        if path.name == ".env.example":
            for line in data.decode().splitlines():
                if line.startswith("OPENAI_API_KEY=") and line.partition("=")[2].strip():
                    errors.append((name, "nonempty-key-example"))
    for name, label in errors:
        print(f"REJECT {name}: {label}")
    if errors:
        return 1
    print("Staged source/privacy checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
