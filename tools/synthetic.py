"""Generate a reproducible, non-private integration clip outside the repository."""
import argparse
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def generate(path, codec="ffv1"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=20)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        stream.time_base = Fraction(1, 20)
        if codec == "libx264":
            stream.options = {"preset": "ultrafast", "tune": "zerolatency", "x264-params": "keyint=20:min-keyint=20:scenecut=0"}
        for index in range(120):
            pixels = np.full((64, 96, 3), 40, np.uint8)
            if 30 <= index < 70:
                x = 4 + (index - 30) % 50
                pixels[20:40, x:x+20] = 210
            if 70 <= index < 90:
                pixels[:] = 40 if index % 2 else 120
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 20)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--codec", choices=("ffv1", "libx264"), default="ffv1")
    a = p.parse_args()
    generate(a.out, a.codec)
