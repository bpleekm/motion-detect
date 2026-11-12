"""Read media once for the portable reference path and export a stable frame index."""
from fractions import Fraction
from pathlib import Path
import time

from .io import SCHEMA, digest, emit, hold, load, write


def source_for_video(path):
    import av
    path = Path(path).resolve()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        tb = Fraction(stream.time_base)
        duration = stream.duration * tb * 1000 if stream.duration is not None else Fraction(container.duration or 0, 1000)
        if duration <= 0:
            raise ValueError("duration metadata missing; provide an explicit source manifest")
        sha = digest(path)
        return {"schema_version": SCHEMA, "case_id": "video-" + sha[:12], "total_ms": round(duration),
                "files": [{"file_id": sha[:20], "sha256": sha, "path": str(path), "stream_index": stream.index,
                           "time_base": [tb.numerator, tb.denominator], "start_us": 0,
                           "source_start_pts": int(stream.start_time or 0), "width": stream.width, "height": stream.height}]}


def run(input_path, out, *, source_manifest=None, sample_ms=200, watermark=(0, 0, 0, 0), person_model=None, person_score=.25):
    import av
    from .core import ReferenceCore
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    path = Path(input_path) if input_path else None
    if source_manifest:
        source = load(source_manifest)
    elif path and path.is_file():
        try:
            source = source_for_video(path)
        except (ValueError, IndexError):
            return hold(out, "source_metadata", "readable video with duration and a video stream")
    else:
        return hold(out, "input_media", "input video or a source manifest")
    if source.get("schema_version") != SCHEMA or not source.get("files") or source.get("total_ms", 0) <= 0:
        raise ValueError("invalid source manifest")
    for item in source["files"]:
        p = Path(item["path"])
        if not p.is_file():
            return hold(out, "input_media", "a source file listed in the manifest")
        if digest(p) != item["sha256"]:
            raise ValueError("source fingerprint changed")
    write(out / "source.json", source)
    began = time.perf_counter()
    core = None
    selected = decoded = 0
    last = None
    files = []
    from .person_client import GpuPersonClient, needed
    person = GpuPersonClient(out, person_model, person_score)
    with (out / "frames.jsonl").open("w") as index, (out / "observations.jsonl").open("w") as observations:
        for item in source["files"]:
            count = chosen = 0
            with av.open(item["path"]) as container:
                stream = container.streams[item["stream_index"]]
                stream.thread_count = 2
                for frame in container.decode(stream):
                    if frame.pts is None:
                        raise ValueError("source frame has no PTS")
                    tb = Fraction(frame.time_base)
                    ms = round(Fraction(item["start_us"], 1000) + (frame.pts - item["source_start_pts"]) * tb * 1000)
                    frame_id = f"{item['file_id']}:s{stream.index}:pts{frame.pts}"
                    ref = {"frame_id": frame_id, "file_id": item["file_id"], "stream_index": stream.index,
                           "pts": frame.pts, "time_base": [tb.numerator, tb.denominator], "time_ms": ms,
                           "width": frame.width, "height": frame.height, "key_frame": bool(frame.key_frame)}
                    emit(index, ref)
                    decoded += 1
                    count += 1
                    if last is not None and ms - last < sample_ms:
                        continue
                    if core is None:
                        core = ReferenceCore(frame.width, frame.height, watermark)
                    evidence = core.process(frame.to_ndarray(format="gray"))
                    if needed(evidence):
                        evidence["person"] = person.detect(lambda: frame.to_ndarray(format="bgr24"))
                    else:
                        evidence["person"] = {"status": "skipped", "backend": "opencl-fp32", "detections": None}
                    selected += 1
                    chosen += 1
                    emit(observations, {"frame_id": frame_id, "time_ms": ms, "seq": selected, **evidence})
                    last = ms
            files.append({"file_id": item["file_id"], "decoded_frames": count, "selected_frames": chosen, "decoder_eof": True})
    if not decoded:
        return hold(out, "decoded_frames", "source containing decodable frames")
    person.close()
    result = {"schema_version": SCHEMA, "status": "complete", "backend": "cpu-reference", "person": person.summary(),
              "precision": "CPU double feedback; not a GPU FP32 performance claim", "sample_ms": sample_ms,
              "decoded_frames": decoded, "selected_frames": selected, "files": files,
              "wall_seconds": time.perf_counter() - began,
              "artifacts": {name: digest(out / name) for name in ("source.json", "frames.jsonl", "observations.jsonl")}}
    write(out / "status.json", result)
    return result
