"""Parent media metadata stays isolated from the private native pixel process."""
import math
import os
from pathlib import Path
import subprocess
import sys

from .app import source_for_video
from .io import digest, hold, load, write


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


def run(input_path, out, build_root, *, source_manifest=None, sample_ms=200, watermark=(0, 0, 0, 0)):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    build_root = Path(build_root).resolve() if build_root else None
    if build_root is None or not (build_root / "build.json").is_file():
        return hold(out, "native_build", "successful build-native output")
    build = load(build_root / "build.json")
    for name, expected in build["modules"].items():
        if digest(build_root / name) != expected:
            raise ValueError("native module changed")
    source = load(source_manifest) if source_manifest else source_for_video(input_path) if input_path and Path(input_path).is_file() else None
    if source is None:
        return hold(out, "input_media", "input video or source manifest")
    for item in source["files"]:
        if not Path(item["path"]).is_file():
            return hold(out, "input_media", "all files in source manifest")
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("source fingerprint changed")
    width, height = source["files"][0]["width"], source["files"][0]["height"]
    write(out / "source.json", source)
    cfg = {"width": width, "height": height, "watermark": list(watermark), "gray_lut": gray_lut(),
           "files": source["files"], "output": str(out), "sample_ms": sample_ms,
           "capacity": math.ceil(source["total_ms"] / sample_ms) + 2,
           "build_root": str(build_root), "kernel_root": str(Path(build["source_root"]) / "kernels")}
    write(out / "native-config.json", cfg)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(Path(build["opencv_prefix"]) / "lib") + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with (out / "worker.log").open("w") as log:
        result = subprocess.run([sys.executable, "-B", "-m", "motion_detect.worker", str(out / "native-config.json")], env=env, stdout=log, stderr=log)
    if result.returncode:
        return hold(out, "native_runtime", "inspect private worker.log; CPU fallback was not selected")
    return load(out / "status.json")
