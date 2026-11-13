import pytest

from motion_detect.person_client import GpuPersonClient, needs_person_detection
from motion_detect.person_gpu import GpuPersonDetector, PersonGpuHold


def test_on_demand_trigger_is_unchanged():
    base = {"warming": False, "has_candidate": True, "mog_pass": False, "motion_pass": False}
    assert needs_person_detection(base)
    assert not needs_person_detection(base | {"warming": True})
    assert not needs_person_detection(base | {"has_candidate": False})
    assert not needs_person_detection(base | {"mog_pass": True})
    assert not needs_person_detection(base | {"motion_pass": True})


def test_missing_model_holds_before_downloading_image(tmp_path):
    client = GpuPersonClient(tmp_path)

    def forbidden():
        raise AssertionError("missing model must not fetch BGR")

    result = client.detect(forbidden)
    assert result["status"] == "hold" and result["code"] == "person_model"
    assert result["detections"] is None
    assert client.summary()["held"] == 1
    client.close()


def test_missing_gpu_model_never_selects_cpu(tmp_path):
    with pytest.raises(PersonGpuHold, match="person_model"):
        GpuPersonDetector(tmp_path / "missing.onnx")


def test_graph_cpu_fallback_is_hold(tmp_path, monkeypatch):
    import numpy as np

    from motion_detect import person_gpu

    model = tmp_path / "fake.onnx"
    model.write_bytes(b"fake network fixture")

    class Device:
        def available(self):
            return True

        def type(self):
            return 4

    class Net:
        target = None

        def setPreferableBackend(self, value):
            pass

        def setPreferableTarget(self, value):
            self.target = value

        def setInput(self, value):
            pass

        def getUnconnectedOutLayersNames(self):
            return []

        def forward(self, names):
            return []

        def dump(self):
            return "Convolution OCV/CPU"

    net = Net()
    monkeypatch.setattr(person_gpu.cv2.ocl, "setUseOpenCL", lambda value: None)
    monkeypatch.setattr(person_gpu.cv2.ocl, "useOpenCL", lambda: True)
    monkeypatch.setattr(person_gpu.cv2.ocl, "Device_getDefault", lambda: Device())
    monkeypatch.setattr(person_gpu.cv2.ocl, "finish", lambda: None)
    monkeypatch.setattr(person_gpu.cv2.dnn, "readNet", lambda path: net)
    values = iter((0, 100))
    monkeypatch.setattr(person_gpu, "render_nanoseconds", lambda: next(values))
    detector = GpuPersonDetector(model)
    with pytest.raises(PersonGpuHold, match="cpu_layer_fallback"):
        detector.detect(np.zeros((32, 32, 3), np.uint8))
    assert net.target == person_gpu.cv2.dnn.DNN_TARGET_OPENCL
