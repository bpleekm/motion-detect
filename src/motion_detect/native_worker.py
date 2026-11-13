"""Private media ABI; intentionally does not import Python cv2 or av."""

import json
import sys
import time
from pathlib import Path

from .artifacts import SCHEMA, file_sha256, read_json, write_json, write_json_line
from .person_client import GpuPersonClient, needs_person_detection


def main() -> None:
    configuration = read_json(sys.argv[1])
    output_dir = Path(configuration["output"])
    kernel_dir = Path(configuration["kernel_root"])
    sys.path.insert(0, configuration["build_root"])
    import _motion_gpu_front
    import _motion_gpu_math

    started_at = time.perf_counter()
    frontend = _motion_gpu_front.Engine(
        json.dumps(configuration), (kernel_dir / "preprocess.cl").read_text()
    )
    motion_spatial_model = _motion_gpu_math.Math(
        frontend.context_handle(),
        frontend.queue_handle(),
        configuration["width"],
        configuration["height"],
        configuration["capacity"],
        configuration["watermark"],
        (kernel_dir / "motion.cl").read_text(),
        (kernel_dir / "spatial.cl").read_text(),
    )
    person_client = GpuPersonClient(
        output_dir, configuration.get("person_model"), configuration.get("person_score", 0.25)
    )
    with (output_dir / "observations.jsonl").open("w") as stream:
        while frontend.next():
            frontend.preprocess()
            evidence = json.loads(
                motion_spatial_model.process(
                    frontend.prepared_handle(),
                    frontend.foreground_handle(),
                    frontend.stride(),
                    frontend.offset(),
                )
            )
            frame_metadata = json.loads(frontend.metadata())
            evidence["person"] = (
                person_client.detect(frontend.bgr)
                if needs_person_detection(evidence)
                else {"status": "skipped", "backend": "opencl-fp32", "detections": None}
            )
            write_json_line(
                stream,
                {
                    "frame_id": frame_metadata["frame_id"],
                    "time_ms": frame_metadata["time_ms"],
                    "seq": frame_metadata["seq"],
                    **evidence,
                },
            )
            frontend.release()
    with (output_dir / "model-trace.jsonl").open("w") as stream:
        for trace_record in json.loads(motion_spatial_model.model_trace()):
            write_json_line(stream, trace_record)
    frontend_metrics = json.loads(frontend.metrics())
    model_metrics = json.loads(motion_spatial_model.metrics())
    person_client.close()
    if (
        frontend_metrics["resident_at_end"]
        or frontend_metrics["resident_peak"] > 8
        or frontend_metrics["model_updates"] != model_metrics["updates"]
    ):
        raise ValueError("native ordering/lease invariant failed")
    if not all(
        source_file["decoder_eof"] and source_file["demux_eof"]
        for source_file in frontend_metrics["files"]
    ):
        raise ValueError("native source did not drain")
    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "backend": "native-gpu",
        "precision": "GPU FP32 noise",
        "decoded_frames": frontend_metrics["decoded_frames"],
        "selected_frames": frontend_metrics["selected_frames"],
        "files": frontend_metrics["files"],
        "wall_seconds": time.perf_counter() - started_at,
        "native": frontend_metrics,
        "gpu_math": model_metrics,
        "person": person_client.summary(),
        "old_python_cv2_imported": "cv2" in sys.modules,
        "old_python_av_imported": "av" in sys.modules,
        "artifacts": {
            name: file_sha256(output_dir / name)
            for name in ("source.json", "frames.jsonl", "observations.jsonl")
        },
    }
    write_json(output_dir / "status.json", result)
    del motion_spatial_model
    del frontend


if __name__ == "__main__":
    main()
