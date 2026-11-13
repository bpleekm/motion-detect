"""Parse detector commands and dispatch to the selected pipeline."""

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="motion-detect")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--input", type=Path)
    run_parser.add_argument("--source-manifest", type=Path)
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--sample-ms", type=int, default=200)
    run_parser.add_argument("--watermark", type=int, nargs=4, default=[0, 0, 0, 0])
    run_parser.add_argument(
        "--backend", choices=("cpu-reference", "native-gpu"), default="cpu-reference"
    )
    run_parser.add_argument("--native-build", type=Path)
    run_parser.add_argument("--person-model", type=Path)
    run_parser.add_argument("--person-score", type=float, default=0.25)
    native_build_parser = commands.add_parser("build-native")
    native_build_parser.add_argument("--out", type=Path, required=True)
    native_build_parser.add_argument("--opencv-prefix", type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    options = parser.parse_args()
    if options.command == "build-native":
        from .native_build import build_native_modules

        print(json.dumps(build_native_modules(options.out, options.opencv_prefix)))
        return
    if options.sample_ms <= 0:
        parser.error("sample-ms must be positive")
    if not 0 < options.person_score < 1:
        parser.error("person-score must be between zero and one")
    if options.backend == "native-gpu":
        from .native_pipeline import run_detection

        result = run_detection(
            options.input,
            options.out,
            options.native_build,
            source_manifest=options.source_manifest,
            sample_ms=options.sample_ms,
            watermark=options.watermark,
            person_model=options.person_model,
            person_score=options.person_score,
        )
    else:
        from .pipeline import run_detection

        result = run_detection(
            options.input,
            options.out,
            source_manifest=options.source_manifest,
            sample_ms=options.sample_ms,
            watermark=options.watermark,
            person_model=options.person_model,
            person_score=options.person_score,
        )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("artifacts", "files")}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
