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
    args = parser.parse_args()
    if args.sample_ms <= 0:
        parser.error("sample-ms must be positive")
    from .app import run
    result = run(args.input, args.out, source_manifest=args.source_manifest, sample_ms=args.sample_ms, watermark=args.watermark)
    print(json.dumps({k: v for k, v in result.items() if k not in ("artifacts", "files")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
