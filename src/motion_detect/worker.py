"""Private media ABI; intentionally does not import Python cv2 or av."""
import json
from pathlib import Path
import sys
import time

from .io import SCHEMA, digest, emit, load, write
from .person_client import GpuPersonClient, needed


def main():
    cfg = load(sys.argv[1])
    out, shaders = Path(cfg["output"]), Path(cfg["kernel_root"])
    sys.path.insert(0, cfg["build_root"])
    import _motion_gpu_front
    import _motion_gpu_math
    began = time.perf_counter()
    engine = _motion_gpu_front.Engine(json.dumps(cfg), (shaders / "preprocess.cl").read_text())
    model = _motion_gpu_math.Math(engine.context_handle(), engine.queue_handle(), cfg["width"], cfg["height"], cfg["capacity"],
                                  cfg["watermark"], (shaders / "motion.cl").read_text(), (shaders / "spatial.cl").read_text())
    person = GpuPersonClient(out, cfg.get("person_model"), cfg.get("person_score", .25))
    with (out / "observations.jsonl").open("w") as stream:
        while engine.next():
            engine.preprocess()
            evidence = json.loads(model.process(engine.prepared_handle(), engine.foreground_handle(), engine.stride(), engine.offset()))
            meta = json.loads(engine.metadata())
            evidence["person"] = person.detect(engine.bgr) if needed(evidence) else {"status": "skipped", "backend": "opencl-fp32", "detections": None}
            emit(stream, {"frame_id": meta["frame_id"], "time_ms": meta["time_ms"], "seq": meta["seq"], **evidence})
            engine.release()
    with (out / "model-trace.jsonl").open("w") as stream:
        for row in json.loads(model.model_trace()):
            emit(stream, row)
    native, math = json.loads(engine.metrics()), json.loads(model.metrics())
    person.close()
    if native["resident_at_end"] or native["resident_peak"] > 8 or native["model_updates"] != math["updates"]:
        raise ValueError("native ordering/lease invariant failed")
    if not all(x["decoder_eof"] and x["demux_eof"] for x in native["files"]):
        raise ValueError("native source did not drain")
    result = {"schema_version": SCHEMA, "status": "complete", "backend": "native-gpu", "precision": "GPU FP32 noise",
              "decoded_frames": native["decoded_frames"], "selected_frames": native["selected_frames"], "files": native["files"],
              "wall_seconds": time.perf_counter() - began, "native": native, "gpu_math": math, "person": person.summary(),
              "old_python_cv2_imported": "cv2" in sys.modules, "old_python_av_imported": "av" in sys.modules,
              "artifacts": {name: digest(out / name) for name in ("source.json", "frames.jsonl", "observations.jsonl")}}
    write(out / "status.json", result)
    del model
    del engine


if __name__ == "__main__":
    main()
