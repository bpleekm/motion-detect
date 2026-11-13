"""Parent media metadata stays isolated from the private native pixel process."""

import math
import os
import subprocess
import sys
from pathlib import Path

from .artifacts import file_sha256, read_json, write_hold_status, write_json
from .pipeline import probe_video_source


def gray_lut():
    import av
    import numpy as np

    frame = av.VideoFrame(256, 16, "yuv420p")
    frame.color_range = 1
    frame.planes[0].update(np.tile(np.arange(256, dtype=np.uint8), (16, 1)))
    for plane in frame.planes[1:]:
        plane.update(bytes([128]) * plane.buffer_size)
    values = frame.to_ndarray(format="gray")
    if not np.array_equal(values, np.broadcast_to(values[0], values.shape)):
        raise ValueError("gray ramp was not row-invariant")
    return values[0].tolist()


def run_detection(
    input_path,
    output_dir,
    build_root,
    *,
    source_manifest=None,
    sample_ms=200,
    watermark=(0, 0, 0, 0),
    person_model=None,
    person_score=0.25,
):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    build_root = Path(build_root).resolve() if build_root else None
    if build_root is None or not (build_root / "build.json").is_file():
        return write_hold_status(output_dir, "native_build", "successful build-native output")
    build_manifest = read_json(build_root / "build.json")
    for name, expected in build_manifest["modules"].items():
        if file_sha256(build_root / name) != expected:
            raise ValueError("native module changed")
    source_manifest_data = (
        read_json(source_manifest)
        if source_manifest
        else probe_video_source(input_path)
        if input_path and Path(input_path).is_file()
        else None
    )
    if source_manifest_data is None:
        return write_hold_status(output_dir, "input_media", "input video or source manifest")
    for source_file in source_manifest_data["files"]:
        if not Path(source_file["path"]).is_file():
            return write_hold_status(output_dir, "input_media", "all files in source manifest")
        if file_sha256(source_file["path"]) != source_file["sha256"]:
            raise ValueError("source fingerprint changed")
    width = source_manifest_data["files"][0]["width"]
    height = source_manifest_data["files"][0]["height"]
    write_json(output_dir / "source.json", source_manifest_data)
    configuration = {
        "width": width,
        "height": height,
        "watermark": list(watermark),
        "gray_lut": gray_lut(),
        "files": source_manifest_data["files"],
        "output": str(output_dir),
        "sample_ms": sample_ms,
        "capacity": math.ceil(source_manifest_data["total_ms"] / sample_ms) + 2,
        "build_root": str(build_root),
        "kernel_root": str(Path(build_manifest["source_root"]) / "kernels"),
        "person_model": str(Path(person_model).resolve()) if person_model else None,
        "person_score": person_score,
    }
    write_json(output_dir / "native-config.json", configuration)
    child_environment = dict(os.environ)
    child_environment["LD_LIBRARY_PATH"] = str(Path(build_manifest["opencv_prefix"]) / "lib") + (
        ":" + child_environment["LD_LIBRARY_PATH"]
        if child_environment.get("LD_LIBRARY_PATH")
        else ""
    )
    child_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with (output_dir / "worker.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "motion_detect.native_worker",
                str(output_dir / "native-config.json"),
            ],
            env=child_environment,
            stdout=log,
            stderr=log,
        )
    if result.returncode:
        return write_hold_status(
            output_dir,
            "native_runtime",
            "inspect private worker.log; CPU fallback was not selected",
        )
    return read_json(output_dir / "status.json")
