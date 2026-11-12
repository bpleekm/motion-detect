"""Lazy on-demand GPU person process; missing features are not 'no person'."""
import json
from pathlib import Path
import subprocess
import sys


def needed(observation):
    return not observation["warming"] and observation["has_candidate"] and not (observation["mog_pass"] or observation["motion_pass"])


class GpuPersonClient:
    def __init__(self, out, model=None, score=.25):
        self.out, self.model, self.score = Path(out), Path(model).resolve() if model else None, score
        self.process = None
        self.log = None
        self.failure = None
        self.requested = self.completed = self.held = 0

    def detect(self, image_supplier):
        self.requested += 1
        if self.failure is None and (self.model is None or not self.model.is_file()):
            self.failure = "person_model"
        if self.failure is None and self.process is None:
            self.log = (self.out / "person-worker.log").open("w")
            self.process = subprocess.Popen([sys.executable, "-B", "-m", "motion_detect.person_worker", "--model", str(self.model), "--score", str(self.score)],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
            try:
                hello = json.loads(self.process.stdout.readline())
                if hello["status"] != "ready":
                    self.failure = hello["code"]
            except (ValueError, KeyError):
                self.failure = "person_worker_start"
        if self.failure is not None:
            self.held += 1
            return {"status": "hold", "code": self.failure, "backend": "opencl-fp32", "detections": None}
        try:
            import numpy as np
            image = np.ascontiguousarray(image_supplier(), dtype=np.uint8)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("invalid BGR image")
            self.process.stdin.write((json.dumps({"shape": list(image.shape), "bytes": image.nbytes}) + "\n").encode())
            self.process.stdin.write(image.data)
            self.process.stdin.flush()
            result = json.loads(self.process.stdout.readline())
        except (ValueError, BrokenPipeError, OSError):
            result = {"status": "hold", "code": "person_worker_runtime", "backend": "opencl-fp32", "detections": None}
        if result["status"] == "complete":
            self.completed += 1
        else:
            self.failure = result["code"]
            self.held += 1
        return result

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
                self.process.wait(timeout=30)
            if self.log:
                self.log.close()

    def summary(self):
        return {"backend": "opencl-fp32", "requested": self.requested, "completed": self.completed,
                "held": self.held, "hold_code": self.failure}
