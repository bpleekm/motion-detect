import argparse
import json
import sys

import numpy as np

from .person_gpu import GpuPersonDetector, PersonGpuHold


def send(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--score", type=float, default=.25)
    args = p.parse_args()
    try:
        detector = GpuPersonDetector(args.model)
    except PersonGpuHold as error:
        send({"status": "hold", "code": str(error)})
        return
    except Exception as error:
        send({"status": "hold", "code": "person_model_init:" + type(error).__name__})
        return
    send({"status": "ready"})
    while line := sys.stdin.buffer.readline():
        request = json.loads(line)
        data = sys.stdin.buffer.read(request["bytes"])
        if len(data) != request["bytes"]:
            break
        image = np.frombuffer(data, dtype=np.uint8).reshape(request["shape"])
        try:
            send(detector.detect(image, args.score))
        except PersonGpuHold as error:
            send({"status": "hold", "code": str(error), "backend": "opencl-fp32", "detections": None})
        except Exception as error:
            send({"status": "hold", "code": "person_gpu_forward:" + type(error).__name__, "backend": "opencl-fp32", "detections": None})


if __name__ == "__main__":
    main()
