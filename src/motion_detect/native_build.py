"""Build only into an explicit external directory, using an explicit private SDK."""

import shlex
import subprocess
import sysconfig
from pathlib import Path

from .artifacts import file_sha256, write_hold_status, write_json


def native_source_root():
    checkout = Path(__file__).resolve().parents[2] / "native"
    return checkout if checkout.is_dir() else Path(__file__).with_name("native")


def build_native_modules(out, opencv_prefix):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cv = Path(opencv_prefix).resolve() if opencv_prefix else None
    if cv is None or not (cv / "include/opencv4/opencv2/core.hpp").is_file():
        return write_hold_status(
            out, "private_opencv_sdk", "OpenCL-enabled private OpenCV core/video SDK prefix"
        )
    import pybind11

    root = native_source_root()
    try:
        flags = shlex.split(
            subprocess.check_output(
                [
                    "pkg-config",
                    "--cflags",
                    "--libs",
                    "libavcodec",
                    "libavformat",
                    "libavutil",
                    "libswscale",
                    "libva",
                    "libva-drm",
                    "OpenCL",
                ],
                text=True,
                stderr=subprocess.PIPE,
            )
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return write_hold_status(
            out,
            "native_development_libraries",
            "FFmpeg, VA and OpenCL development packages/pkg-config",
        )
    common = [
        "g++",
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-pthread",
        "-ffp-contract=off",
        "-I" + sysconfig.get_paths()["include"],
        "-I" + pybind11.get_include(),
    ]
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    commands = {
        "front": common
        + [
            "-I" + str(cv / "include/opencv4"),
            str(root / "video_frontend.cpp"),
            "-L" + str(cv / "lib"),
            "-Wl,-rpath," + str(cv / "lib"),
            "-lopencv_video",
            "-lopencv_core",
            *flags,
            "-o",
            str(out / ("_motion_gpu_front" + suffix)),
        ],
        "math": common
        + [
            str(root / "motion_spatial.cpp"),
            "-lOpenCL",
            "-o",
            str(out / ("_motion_gpu_math" + suffix)),
        ],
    }
    for name, command in commands.items():
        result = subprocess.run(command, text=True, capture_output=True)
        write_json(
            out / (name + "-compile.json"),
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        if result.returncode:
            return write_hold_status(
                out, "native_compile", "inspect private compile receipt and resolve build error"
            )
    metadata = {
        "status": "complete",
        "opencv_prefix": str(cv),
        "source_root": str(root),
        "sources": {
            str(p.relative_to(root)): file_sha256(p) for p in root.rglob("*") if p.is_file()
        },
        "modules": {p.name: file_sha256(p) for p in out.glob("*.so")},
    }
    write_json(out / "build.json", metadata)
    return {"status": "complete", "modules": len(metadata["modules"])}
