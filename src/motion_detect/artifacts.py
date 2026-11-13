"""Small versioned artifacts; all output paths are supplied by the caller."""

import json
from hashlib import file_digest
from pathlib import Path

SCHEMA = "motion-detect/v1"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_json_lines(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_hold_status(output_dir, code, requirement):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": SCHEMA,
        "status": "hold",
        "missing": [{"code": code, "requirement": requirement}],
    }
    write_json(output_dir / "status.json", value)
    return value
