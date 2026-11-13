"""Read media once for the portable reference path and export a stable frame index."""

import time
from fractions import Fraction
from pathlib import Path

from .artifacts import (
    SCHEMA,
    file_sha256,
    read_json,
    write_hold_status,
    write_json,
    write_json_line,
)


def probe_video_source(path):
    import av

    path = Path(path).resolve()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = Fraction(stream.time_base)
        duration = (
            stream.duration * time_base * 1000
            if stream.duration is not None
            else Fraction(container.duration or 0, 1000)
        )
        if duration <= 0:
            raise ValueError("duration metadata missing; provide an explicit source manifest")
        source_sha256 = file_sha256(path)
        return {
            "schema_version": SCHEMA,
            "case_id": "video-" + source_sha256[:12],
            "total_ms": round(duration),
            "files": [
                {
                    "file_id": source_sha256[:20],
                    "sha256": source_sha256,
                    "path": str(path),
                    "stream_index": stream.index,
                    "time_base": [time_base.numerator, time_base.denominator],
                    "start_us": 0,
                    "source_start_pts": int(stream.start_time or 0),
                    "width": stream.width,
                    "height": stream.height,
                }
            ],
        }


def run_detection(
    input_path,
    output_dir,
    *,
    source_manifest=None,
    sample_ms=200,
    watermark=(0, 0, 0, 0),
    person_model=None,
    person_score=0.25,
):
    import av

    from .core import ReferenceCore

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    path = Path(input_path) if input_path else None
    if source_manifest:
        source = read_json(source_manifest)
    elif path and path.is_file():
        try:
            source = probe_video_source(path)
        except (ValueError, IndexError):
            return write_hold_status(
                output_dir, "source_metadata", "readable video with duration and a video stream"
            )
    else:
        return write_hold_status(output_dir, "input_media", "input video or a source manifest")
    if (
        source.get("schema_version") != SCHEMA
        or not source.get("files")
        or source.get("total_ms", 0) <= 0
    ):
        raise ValueError("invalid source manifest")
    for source_file in source["files"]:
        source_path = Path(source_file["path"])
        if not source_path.is_file():
            return write_hold_status(
                output_dir, "input_media", "a source file listed in the manifest"
            )
        if file_sha256(source_path) != source_file["sha256"]:
            raise ValueError("source fingerprint changed")
    write_json(output_dir / "source.json", source)
    started_at = time.perf_counter()
    core = None
    selected = decoded = 0
    last_selected_ms = None
    file_results = []
    from .person_client import GpuPersonClient, needs_person_detection

    person_client = GpuPersonClient(output_dir, person_model, person_score)
    with (
        (output_dir / "frames.jsonl").open("w") as index,
        (output_dir / "observations.jsonl").open("w") as observations,
    ):
        for source_file in source["files"]:
            count = chosen = 0
            with av.open(source_file["path"]) as container:
                stream = container.streams[source_file["stream_index"]]
                stream.thread_count = 2
                for frame in container.decode(stream):
                    if frame.pts is None:
                        raise ValueError("source frame has no PTS")
                    time_base = Fraction(frame.time_base)
                    time_ms = round(
                        Fraction(source_file["start_us"], 1000)
                        + (frame.pts - source_file["source_start_pts"]) * time_base * 1000
                    )
                    frame_id = f"{source_file['file_id']}:s{stream.index}:pts{frame.pts}"
                    frame_record = {
                        "frame_id": frame_id,
                        "file_id": source_file["file_id"],
                        "stream_index": stream.index,
                        "pts": frame.pts,
                        "time_base": [time_base.numerator, time_base.denominator],
                        "time_ms": time_ms,
                        "width": frame.width,
                        "height": frame.height,
                        "key_frame": bool(frame.key_frame),
                    }
                    write_json_line(index, frame_record)
                    decoded += 1
                    count += 1
                    if last_selected_ms is not None and time_ms - last_selected_ms < sample_ms:
                        continue
                    if core is None:
                        core = ReferenceCore(frame.width, frame.height, watermark)
                    evidence = core.process(frame.to_ndarray(format="gray"))
                    if needs_person_detection(evidence):
                        evidence["person"] = person_client.detect(
                            lambda: frame.to_ndarray(format="bgr24")
                        )
                    else:
                        evidence["person"] = {
                            "status": "skipped",
                            "backend": "opencl-fp32",
                            "detections": None,
                        }
                    selected += 1
                    chosen += 1
                    write_json_line(
                        observations,
                        {"frame_id": frame_id, "time_ms": time_ms, "seq": selected, **evidence},
                    )
                    last_selected_ms = time_ms
            file_results.append(
                {
                    "file_id": source_file["file_id"],
                    "decoded_frames": count,
                    "selected_frames": chosen,
                    "decoder_eof": True,
                }
            )
    if not decoded:
        return write_hold_status(output_dir, "decoded_frames", "source containing decodable frames")
    person_client.close()
    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "backend": "cpu-reference",
        "person": person_client.summary(),
        "precision": "CPU double feedback; not a GPU FP32 performance claim",
        "sample_ms": sample_ms,
        "decoded_frames": decoded,
        "selected_frames": selected,
        "files": file_results,
        "wall_seconds": time.perf_counter() - started_at,
        "artifacts": {
            name: file_sha256(output_dir / name)
            for name in ("source.json", "frames.jsonl", "observations.jsonl")
        },
    }
    write_json(output_dir / "status.json", result)
    return result
