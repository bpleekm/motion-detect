from pathlib import Path
import importlib.util

import numpy as np

from motion_detect.app import run
from motion_detect.core import ReferenceCore, components, denoise
from motion_detect.io import load, rows


def synthetic(path):
    script = Path(__file__).parents[1] / "tools/synthetic.py"
    spec = importlib.util.spec_from_file_location("synthetic_clip", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate(path)


def test_complete_identity_and_eof(tmp_path):
    video = synthetic(tmp_path / "source.mkv")
    out = tmp_path / "detect"
    status = run(video, out)
    index = list(rows(out / "frames.jsonl"))
    selected = list(rows(out / "observations.jsonl"))
    assert status["status"] == "complete"
    assert status["decoded_frames"] == len(index) == 120
    assert status["selected_frames"] == len(selected) == 30
    assert len({x["frame_id"] for x in index}) == 120
    assert [x["time_ms"] for x in index] == list(range(0, 6000, 50))
    assert all(b["time_ms"] - a["time_ms"] >= 200 for a, b in zip(selected, selected[1:]))
    assert status["files"][0]["decoder_eof"]
    assert load(out / "source.json")["total_ms"] == 6000


def test_missing_media_is_hold(tmp_path):
    status = run(tmp_path / "missing.mkv", tmp_path / "detect")
    assert status["status"] == "hold"
    assert status["missing"][0]["code"] == "input_media"


def test_four_connectivity_and_area_equality():
    mask = np.zeros((16, 16), np.uint8)
    mask[3, :5] = 255
    mask[4, 5:10] = 255
    kept, regions = components(mask, 5)
    assert [r["area"] for r in regions] == [5, 5]
    assert np.array_equal(kept, mask)
    kept, regions = components(mask, 5, True)
    assert not regions and not kept.any()


def test_model_order_and_independent_opening():
    core = ReferenceCore(64, 64)
    image = np.full((64, 64), 40, np.uint8)
    first = core.process(image)
    assert not first["has_candidate"] and first["warming"]
    for _ in range(5):
        result = core.process(image)
    assert not result["warming"]
    assert core.updates == 6 and np.array_equal(core.reference, image)
    raw = np.zeros((64, 64), np.uint8)
    raw[0:20, 0:20] = 255
    assert not np.any(denoise(raw, 2) & np.uint8(raw == 0))
