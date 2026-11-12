"""Small versioned artifacts; all output paths are supplied by the caller."""
from hashlib import file_digest
import json
from pathlib import Path

SCHEMA = "motion-detect/v1"


def digest(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def emit(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def hold(out, code, requirement):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": SCHEMA, "status": "hold", "missing": [{"code": code, "requirement": requirement}]}
    write(out / "status.json", value)
    return value
