import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="motion-detect")
    sub = parser.add_subparsers(dest="command", required=True)
    detect = sub.add_parser("run")
    detect.add_argument("--input", type=Path)
    detect.add_argument("--source-manifest", type=Path)
    detect.add_argument("--out", type=Path, required=True)
    detect.add_argument("--sample-ms", type=int, default=200)
    detect.add_argument("--watermark", type=int, nargs=4, default=[0, 0, 0, 0])
    detect.add_argument(
        "--backend", choices=("cpu-reference", "native-gpu"), default="cpu-reference"
    )
    detect.add_argument("--native-build", type=Path)
    detect.add_argument("--person-model", type=Path)
    detect.add_argument("--person-score", type=float, default=0.25)
    build = sub.add_parser("build-native")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--opencv-prefix", type=Path)
    args = parser.parse_args()
    if args.command == "build-native":
        from .native_build import build

        print(json.dumps(build(args.out, args.opencv_prefix)))
        return
    if args.sample_ms <= 0:
        parser.error("sample-ms must be positive")
    if not 0 < args.person_score < 1:
        parser.error("person-score must be between zero and one")
    if args.backend == "native-gpu":
        from .native_app import run

        result = run(
            args.input,
            args.out,
            args.native_build,
            source_manifest=args.source_manifest,
            sample_ms=args.sample_ms,
            watermark=args.watermark,
            person_model=args.person_model,
            person_score=args.person_score,
        )
    else:
        from .app import run

        result = run(
            args.input,
            args.out,
            source_manifest=args.source_manifest,
            sample_ms=args.sample_ms,
            watermark=args.watermark,
            person_model=args.person_model,
            person_score=args.person_score,
        )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("artifacts", "files")}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
