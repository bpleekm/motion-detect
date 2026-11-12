"""Original NanoDet image/postprocessing with an explicitly observed OpenCL forward."""
import math
from hashlib import file_digest
from pathlib import Path

import cv2
import numpy as np


class PersonGpuHold(RuntimeError):
    pass


def render_nanoseconds():
    clients = {}
    for path in Path("/proc/self/fdinfo").iterdir():
        try:
            fields = dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
        except FileNotFoundError:
            continue
        if fields.get("drm-driver", "").strip() not in ("i915", "xe"):
            continue
        clients[fields.get("drm-client-id", "").strip()] = int(fields.get("drm-engine-render", "0").strip().split()[0])
    return sum(clients.values())


class GpuPersonDetector:
    def __init__(self, model):
        if not Path(model).is_file():
            raise PersonGpuHold("person_model")
        with Path(model).open("rb") as stream:
            self.model_sha256 = file_digest(stream, "sha256").hexdigest()
        cv2.setNumThreads(2)
        cv2.ocl.setUseOpenCL(True)
        device = cv2.ocl.Device_getDefault()
        if not cv2.ocl.useOpenCL() or not device.available() or not device.type() & 4:
            raise PersonGpuHold("person_gpu_device")
        self.net = cv2.dnn.readNet(str(model))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        self.anchors = []
        for stride in (8, 16, 32, 64):
            side = 416 // stride
            xx, yy = np.meshgrid(np.arange(side) * stride, np.arange(side) * stride)
            self.anchors.append(np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32) + .5 * (stride - 1))
        self.graph_checked = False

    def detect(self, bgr, threshold=.25):
        height, width = bgr.shape[:2]
        scale = min(416 / width, 416 / height)
        rw, rh = round(width * scale), round(height * scale)
        xp, yp = (416 - rw) // 2, (416 - rh) // 2
        canvas = np.zeros((416, 416, 3), np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        canvas[yp:yp+rh, xp:xp+rw] = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)
        values = (canvas.astype(np.float32) - np.array([103.53, 116.28, 123.675], np.float32)) / np.array([57.375, 57.12, 58.395], np.float32)
        self.net.setInput(cv2.dnn.blobFromImage(values))
        cv2.ocl.finish()
        before = render_nanoseconds()
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        cv2.ocl.finish()
        delta = render_nanoseconds() - before
        if delta <= 0:
            raise PersonGpuHold("person_gpu_execution_unproven")
        if not self.graph_checked:
            graph = self.net.dump()
            if "OCV/CPU" in graph or "OCV/OCL" not in graph:
                raise PersonGpuHold("person_gpu_cpu_layer_fallback")
            self.graph_checked = True
        return self._postprocess(outputs, width, height, scale, xp, yp, threshold, delta)

    def _postprocess(self, outputs, width, height, scale, xp, yp, threshold, delta):
        boxes, scores = [], []
        for stride, anchors, classification, regression in zip((8, 16, 32, 64), self.anchors, outputs[::2], outputs[1::2]):
            classification, regression = classification.reshape(-1, 80), regression.reshape(-1, 4, 8)
            if len(classification) > 1000:
                order = np.argsort(classification.max(axis=1))[-1000:]
                classification, regression, anchors = classification[order], regression[order], anchors[order]
            selected = (classification.argmax(axis=1) == 0) & (classification[:, 0] >= threshold)
            if not np.any(selected):
                continue
            classification, regression, anchors = classification[selected], regression[selected], anchors[selected]
            values = np.exp(regression - regression.max(axis=2, keepdims=True))
            distance = ((values / values.sum(axis=2, keepdims=True)) * np.arange(8)).sum(axis=2) * stride
            corners = np.column_stack((anchors[:, 0] - distance[:, 0], anchors[:, 1] - distance[:, 1], anchors[:, 0] + distance[:, 2], anchors[:, 1] + distance[:, 3]))
            for corner, score in zip(corners, classification[:, 0]):
                x0, y0 = max(0, math.floor((corner[0] - xp) / scale)), max(0, math.floor((corner[1] - yp) / scale))
                x1, y1 = min(width, math.ceil((corner[2] - xp) / scale)), min(height, math.ceil((corner[3] - yp) / scale))
                if x1 > x0 and y1 > y0:
                    boxes.append([x0, y0, x1 - x0, y1 - y0])
                    scores.append(float(score))
        keep = np.asarray(cv2.dnn.NMSBoxes(boxes, scores, threshold, .6)).reshape(-1) if boxes else []
        detections = [{"box": boxes[int(i)], "score": round(scores[int(i)], 4)} for i in sorted(keep, key=lambda i: scores[int(i)], reverse=True)]
        return {"status": "complete", "backend": "opencl-fp32", "detections": detections,
                "gpu_render_ns": delta, "cpu_layer_fallback": False, "model_sha256": self.model_sha256,
                "pre_postprocessing": "CPU original blob/NMS"}
