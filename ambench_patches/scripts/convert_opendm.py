#!/usr/bin/env python
"""LeRobot (v2.1, AM-Bench export) -> OpenDM (DM0.5) JSONL converter.

Reads a LeRobot dataset root that stores images as PNG bytes inside the
parquet files (column ``ee_image``) and writes the JSONL layout consumed by
``opendm/data/dataset.py::JsonlDataset`` + ``opendm/data/transforms.py``:

    <out>/
      dataset.json                     manifest (fps, dims, tasks, counts, source)
      episodes.jsonl                   one line per converted episode
      jsonl/index_cache.json           {"data": {"<abs jsonl path>": <n_rows>}}
      jsonl/<Task>/episode_XXXXXX.jsonl   one episode per file, one frame per line
      images/<Task>/episode_XXXXXX/frame_XXXXXX.png
      images_base/<Task>/episode_XXXXXX/frame_XXXXXX.png   (only with --base-image-col)

Two-camera mode (``--base-image-col base_image``): every record additionally has
``images_2`` (base camera). OpenDM's ``LoadImages`` resolves every url as
``os.path.join(image_dir, url)``, so to keep the two camera trees as siblings
(``images/`` and ``images_base/``) the manifest's ``image_dir`` becomes the
dataset ROOT and both urls carry their top-level directory
(``images/<Task>/...`` and ``images_base/<Task>/...``). Without the flag the
single-camera layout/urls are byte-for-byte what they always were.

Per-frame record (keys read by OpenDM: images_1 / state / action / prompt):

    {"images_1": {"type": "image", "url": "<Task>/episode_XXXXXX/frame_XXXXXX.png"},
     "state": [x, y, z, ax, ay, az, gripper],
     "action": [x, y, z, ax, ay, az, gripper],
     "prompt": "press the button", "is_robot": true, "task_name": "ambench_PressButton",
     "episode_index": 12, "frame_index": 3, "timestamp": 0.15}

Conventions
- state  = ee_pos(3) + quat_wxyz_to_axis_angle(ee_quat)(3) + gripper_width(1)
- action = actions[:3] + quat_wxyz_to_axis_angle(actions[3:7])(3) + actions[7]
  (``actions`` is the ABSOLUTE end-effector target pose in the world frame; no
  differencing is done here, OpenDM's ``ActionMode`` decides relative/absolute).
- quat -> axis-angle mirrors ``opendm.data.transforms._quat_to_rotvec``:
  normalize, flip sign so that w >= 0, angle = 2*atan2(|v|, w) in [0, pi].
- gripper: state uses ``gripper_width`` in metres (>= 0); action uses the raw
  ``actions[7]`` gripper COMMAND which in the AM-Bench export is normalized to
  [-1, 1] (not metres). Both are kept as-is; OpenDM normalizes state and action
  with separate statistics and keeps the gripper dimension absolute in RELATIVE
  mode, so the unit mismatch is harmless for training and the model emits the
  command the controller expects.
- This matches ``RobotStateDesc = [EEF]*6 + [GRIPPER]`` (same as RobotType.UR5).

Idempotent: re-running overwrites the selected episodes; generated
``episode_*.jsonl`` files (and their image folders) that are not part of the
current selection are removed so that ``jsonl_dir`` always equals the index.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

# Human alias (CamelCase) <-> instruction string found in meta/tasks.jsonl.
TASK_ALIASES = {
    "press the button": "PressButton",
    "insert the peg into the hole": "PegInHole",
    "push the slider to the right": "PushSlider",
    "open the door": "OpenDoor",
    "pick the yellow lemon and place it in the bin": "LemonHarvesting",
    "pull the lever upwards": "PullLever",
    "rotate the valve clockwise": "RotateValve",
    "contact the inspection point and hold": "NDT",
    "open the cabinet sliding door, pick up the red can, and place it on top of the cabinet": "CabinetPickPlace",
}
ALIAS_TO_TASK = {v: k for k, v in TASK_ALIASES.items()}

STATE_DESC = ["eef", "eef", "eef", "eef", "eef", "eef", "gripper"]
STATE_NAMES = ["x", "y", "z", "ax", "ay", "az", "gripper"]
PNG_SIG = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- #
# math helpers
# --------------------------------------------------------------------------- #
def quat_wxyz_to_axis_angle(q) -> np.ndarray:
    """Quaternion (w, x, y, z) -> rotation vector, |rotvec| <= pi.

    Same algorithm as opendm.data.transforms._quat_to_rotvec so that the
    training-side representation and the inference-side reconstruction agree.
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-8:
        raise ValueError(f"degenerate quaternion {q}")
    q = q / n
    if q[0] < 0:
        q = -q
    vec = q[1:4]
    vn = np.linalg.norm(vec)
    angle = 2.0 * math.atan2(vn, q[0])
    if vn < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return vec * (angle / vn)


def pose8_to_7(x) -> list[float]:
    """[x,y,z, qw,qx,qy,qz, g] -> [x,y,z, ax,ay,az, g]."""
    x = np.asarray(x, dtype=np.float64).reshape(8)
    aa = quat_wxyz_to_axis_angle(x[3:7])
    out = np.concatenate([x[:3], aa, x[7:8]])
    if not np.all(np.isfinite(out)):
        raise ValueError(f"non-finite pose {x}")
    return [float(v) for v in out]


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def task_dirname(task: str) -> str:
    return TASK_ALIASES.get(task, slug(task))


def resolve_task_filter(tasks_arg, known_tasks):
    """Map user supplied names (instruction or CamelCase alias) to instruction strings."""
    if not tasks_arg:
        return None
    selected = []
    lower_known = {t.lower(): t for t in known_tasks}
    for name in tasks_arg:
        n = name.strip()
        if n in known_tasks:
            selected.append(n)
        elif n.lower() in lower_known:
            selected.append(lower_known[n.lower()])
        elif n in ALIAS_TO_TASK and ALIAS_TO_TASK[n] in known_tasks:
            selected.append(ALIAS_TO_TASK[n])
        else:
            # alias may also match a slug of an unknown instruction
            hit = [t for t in known_tasks if slug(t).lower() == slug(n).lower()]
            if len(hit) == 1:
                selected.append(hit[0])
            else:
                raise SystemExit(
                    f"unknown task {name!r}; known: {sorted(known_tasks)} "
                    f"aliases: {sorted(ALIAS_TO_TASK)}"
                )
    # keep order, drop duplicates
    seen, out = set(), []
    for t in selected:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #
def image_cell_to_bytes(cell, src_root: Path) -> bytes:
    """Return encoded image bytes from a LeRobot/HF ``Image`` parquet cell."""
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if b is not None:
            return bytes(b)
        p = cell.get("path")
        if p:
            fp = Path(p)
            if not fp.is_absolute():
                fp = src_root / fp
            return fp.read_bytes()
        raise ValueError("image cell dict has neither bytes nor path")
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return bytes(cell)
    if isinstance(cell, np.ndarray):
        from PIL import Image

        arr = cell
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()
    raise TypeError(f"unsupported image cell type {type(cell)}")


def write_image(raw: bytes, dst: Path, fmt: str, jpg_quality: int, expect_hw):
    """Write one frame. PNG source + png target -> byte copy (lossless, fast)."""
    from PIL import Image

    im = Image.open(io.BytesIO(raw))  # lazy: header only
    if expect_hw is not None and (im.size[1], im.size[0]) != tuple(expect_hw):
        raise ValueError(f"unexpected image size {im.size} (w,h), expected h,w={expect_hw}")
    if fmt == "png" and raw[:8] == PNG_SIG and im.mode == "RGB":
        dst.write_bytes(raw)
        return
    im = im.convert("RGB")
    if fmt == "png":
        im.save(dst, format="PNG", compress_level=6)
    else:
        im.save(dst, format="JPEG", quality=jpg_quality, subsampling=0)


# --------------------------------------------------------------------------- #
# per-episode worker
# --------------------------------------------------------------------------- #
def convert_episode(job: dict) -> dict:
    import pyarrow.parquet as pq

    t0 = time.time()
    src_root = Path(job["src_root"])
    parquet = src_root / job["parquet_rel"]
    ep = int(job["episode_index"])
    task = job["task"]
    tdir = job["task_dir"]
    fmt = job["image_format"]
    ext = "png" if fmt == "png" else "jpg"
    fps = float(job["fps"])
    expect_hw = job.get("expect_hw")
    base_col = job.get("base_image_col") or ""   # "" -> single camera (original behaviour)

    tbl = pq.read_table(parquet)
    cols = set(tbl.column_names)
    need = {"ee_image", "actions", "ee_pos", "ee_quat", "gripper_width", "frame_index"}
    if base_col:
        need = need | {base_col}
    missing = need - cols
    if missing:
        raise ValueError(f"{parquet}: missing columns {sorted(missing)}")

    frame_index = np.asarray(tbl.column("frame_index").to_pylist(), dtype=np.int64)
    n = len(frame_index)
    if n < 2:
        raise ValueError(f"episode {ep} has {n} frames (<2)")
    if "episode_index" in cols:
        epi = np.asarray(tbl.column("episode_index").to_pylist())
        if not np.all(epi == ep):
            raise ValueError(f"{parquet}: episode_index column != {ep}")
    if not np.array_equal(frame_index, np.arange(n)):
        raise ValueError(f"{parquet}: frame_index not contiguous 0..{n-1}")
    if "task_index" in cols:
        ti = np.asarray(tbl.column("task_index").to_pylist())
        if not np.all(ti == job["task_index"]):
            raise ValueError(f"{parquet}: task_index column != {job['task_index']}")

    actions = np.asarray(tbl.column("actions").to_pylist(), dtype=np.float64).reshape(n, 8)
    ee_pos = np.asarray(tbl.column("ee_pos").to_pylist(), dtype=np.float64).reshape(n, 3)
    ee_quat = np.asarray(tbl.column("ee_quat").to_pylist(), dtype=np.float64).reshape(n, 4)
    grip = np.asarray(tbl.column("gripper_width").to_pylist(), dtype=np.float64).reshape(n, 1)
    if "timestamp" in cols:
        ts = np.asarray(tbl.column("timestamp").to_pylist(), dtype=np.float64).reshape(n)
    else:
        ts = frame_index / fps
    ts_warn = float(np.max(np.abs(ts - frame_index / fps)))
    images = tbl.column("ee_image").to_pylist()
    images_base = tbl.column(base_col).to_pylist() if base_col else None
    del tbl

    out_root = Path(job["out_root"])
    img_dir = out_root / "images" / tdir / f"episode_{ep:06d}"
    img_base_dir = out_root / "images_base" / tdir / f"episode_{ep:06d}"
    jsonl_path = out_root / "jsonl" / tdir / f"episode_{ep:06d}.jsonl"
    # single camera: url relative to <out>/images ; two cameras: url relative to <out> (see module doc)
    url1_prefix = "images/" if base_col else ""
    # idempotent: wipe stale frames of this episode
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    if base_col:
        if img_base_dir.exists():
            shutil.rmtree(img_base_dir)
        img_base_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    max_aa_state = 0.0
    max_aa_action = 0.0
    g_state = [float(grip.min()), float(grip.max())]
    g_action = [float(actions[:, 7].min()), float(actions[:, 7].max())]
    tmp = jsonl_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for i in range(n):
            state = pose8_to_7(np.concatenate([ee_pos[i], ee_quat[i], grip[i]]))
            action = pose8_to_7(actions[i])
            max_aa_state = max(max_aa_state, float(np.linalg.norm(state[3:6])))
            max_aa_action = max(max_aa_action, float(np.linalg.norm(action[3:6])))
            fname = f"frame_{i:06d}.{ext}"
            raw = image_cell_to_bytes(images[i], src_root)
            write_image(raw, img_dir / fname, fmt, job["jpg_quality"], expect_hw)
            rec = {
                "images_1": {"type": "image", "url": f"{url1_prefix}{tdir}/episode_{ep:06d}/{fname}"},
            }
            if base_col:
                raw_b = image_cell_to_bytes(images_base[i], src_root)
                write_image(raw_b, img_base_dir / fname, fmt, job["jpg_quality"], expect_hw)
                rec["images_2"] = {"type": "image", "url": f"images_base/{tdir}/episode_{ep:06d}/{fname}"}
            rec.update({
                "state": state,
                "action": action,
                "prompt": task,
                "is_robot": True,
                "task_name": f"ambench_{tdir}",
                "episode_index": ep,
                "frame_index": i,
                "timestamp": round(float(ts[i]), 6),
            })
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    os.replace(tmp, jsonl_path)
    img_bytes = sum(p.stat().st_size for p in img_dir.iterdir())
    img_base_bytes = sum(p.stat().st_size for p in img_base_dir.iterdir()) if base_col else 0
    return {
        "episode_index": ep,
        "task": task,
        "task_index": int(job["task_index"]),
        "task_dir": tdir,
        "frames": n,
        "jsonl_abs": str(jsonl_path.resolve()),
        "jsonl_rel": str(jsonl_path.relative_to(out_root)),
        "image_dir_rel": str(img_dir.relative_to(out_root)),
        "image_base_dir_rel": str(img_base_dir.relative_to(out_root)) if base_col else None,
        "image_bytes": int(img_bytes),
        "image_base_bytes": int(img_base_bytes),
        "jsonl_bytes": int(jsonl_path.stat().st_size),
        "max_axis_angle_state": max_aa_state,
        "max_axis_angle_action": max_aa_action,
        "gripper_state_range": g_state,
        "gripper_action_range": g_action,
        "timestamp_max_dev": ts_warn,
        "seconds": time.time() - t0,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def read_jsonl(p: Path):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="LeRobot dataset root (contains meta/ and data/)")
    ap.add_argument("--out", required=True, help="output root for the OpenDM dataset")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="task filter: instruction strings from meta/tasks.jsonl or CamelCase aliases "
                         "(PressButton, PegInHole, PushSlider, OpenDoor, LemonHarvesting, PullLever, RotateValve, NDT)")
    ap.add_argument("--max-episodes", type=int, default=None, help="max episodes PER TASK (after filtering)")
    ap.add_argument("--image-format", choices=["png", "jpg"], default="png")
    ap.add_argument("--jpg-quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=None, help="processes (default: SLURM_CPUS_PER_TASK or 4)")
    ap.add_argument("--keep-stale", action="store_true",
                    help="do not delete previously generated episodes that are outside the current selection")
    ap.add_argument("--base-image-col", default="",
                    help="second (base) camera column, e.g. base_image. When set, each record also gets images_2 "
                         "(files under <out>/images_base/<Task>/...), image_keys=[images_1, images_2] and the "
                         "manifest image_dir is the dataset root. Empty (default) = single-camera behaviour")
    args = ap.parse_args(argv)
    base_col = args.base_image_col or ""

    t_start = time.time()
    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    info = json.loads((src / "meta" / "info.json").read_text())
    tasks_meta = read_jsonl(src / "meta" / "tasks.jsonl")
    episodes_meta = read_jsonl(src / "meta" / "episodes.jsonl")
    fps = info.get("fps", 20)
    if "ee_image" not in info.get("features", {}):
        raise SystemExit("source has no ee_image feature; converter expects the AM-Bench EE-camera export")
    shape = info["features"]["ee_image"].get("shape")  # [C,H,W]
    expect_hw = [int(shape[1]), int(shape[2])] if shape and len(shape) == 3 else None
    if info["features"]["ee_image"].get("dtype") == "video":
        raise SystemExit("ee_image is a video feature; this converter handles image-in-parquet exports only")
    if base_col:
        bf = info.get("features", {}).get(base_col)
        if bf is None:
            raise SystemExit(f"source has no {base_col!r} feature (--base-image-col)")
        if bf.get("dtype") == "video":
            raise SystemExit(f"{base_col} is a video feature; image-in-parquet exports only")
        if bf.get("shape") != info["features"]["ee_image"].get("shape"):
            raise SystemExit(f"{base_col} shape {bf.get('shape')} != ee_image shape {shape}")

    task_by_index = {int(t["task_index"]): t["task"] for t in tasks_meta}
    index_by_task = {v: k for k, v in task_by_index.items()}
    selected_tasks = resolve_task_filter(args.tasks, list(index_by_task))
    if selected_tasks is None:
        selected_tasks = [task_by_index[k] for k in sorted(task_by_index)]

    data_path_tpl = info["data_path"]
    chunks_size = int(info.get("chunks_size", 1000))
    jobs, per_task = [], {}
    for e in episodes_meta:
        ep = int(e["episode_index"])
        etask = e["tasks"][0] if isinstance(e.get("tasks"), list) else e.get("task")
        if etask not in selected_tasks:
            continue
        if args.max_episodes is not None and per_task.get(etask, 0) >= args.max_episodes:
            continue
        per_task[etask] = per_task.get(etask, 0) + 1
        jobs.append({
            "src_root": str(src),
            "out_root": str(out),
            "parquet_rel": data_path_tpl.format(episode_chunk=ep // chunks_size, episode_index=ep),
            "episode_index": ep,
            "task": etask,
            "task_index": index_by_task[etask],
            "task_dir": task_dirname(etask),
            "image_format": args.image_format,
            "jpg_quality": args.jpg_quality,
            "fps": fps,
            "expect_hw": expect_hw,
            "expected_length": int(e.get("length", -1)),
            "base_image_col": base_col,
        })
    if not jobs:
        raise SystemExit("no episodes selected")
    for j in jobs:
        if not (src / j["parquet_rel"]).is_file():
            raise SystemExit(f"missing parquet: {src / j['parquet_rel']}")

    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "4") or 4)
    workers = max(1, min(workers, len(jobs)))
    print(f"[convert] src={src}\n[convert] out={out}\n[convert] tasks={selected_tasks}\n"
          f"[convert] episodes={len(jobs)} per_task={per_task} fps={fps} image_format={args.image_format} "
          f"expect_hw={expect_hw} workers={workers} base_image_col={base_col!r}", flush=True)
    (out / "jsonl").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    if base_col:
        (out / "images_base").mkdir(parents=True, exist_ok=True)

    results = []
    ctx = get_context("spawn")
    done = 0
    total = len(jobs)
    with ctx.Pool(workers) as pool:
        for r in pool.imap_unordered(convert_episode, jobs, chunksize=1):
            done += 1
            results.append(r)
            el = time.time() - t_start
            rate = done / el if el > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else float("nan")
            print(f"[{done}/{total}] ep={r['episode_index']:06d} {r['task_dir']:<16s} frames={r['frames']:4d} "
                  f"img={r['image_bytes']/1e6:6.1f}MB"
                  + (f" base={r['image_base_bytes']/1e6:6.1f}MB" if base_col else "")
                  + f" {r['seconds']:5.1f}s  elapsed={el:7.1f}s eta={eta:7.1f}s",
                  flush=True)
    results.sort(key=lambda r: r["episode_index"])

    # length cross-check with meta/episodes.jsonl
    exp_len = {j["episode_index"]: j["expected_length"] for j in jobs}
    bad = [(r["episode_index"], r["frames"], exp_len[r["episode_index"]])
           for r in results if exp_len[r["episode_index"]] not in (-1, r["frames"])]
    if bad:
        raise SystemExit(f"frame count mismatch vs meta/episodes.jsonl: {bad[:5]}")

    # index_cache.json: keys are absolute jsonl paths (XPolicyLab convention),
    # matching what megfile.smart_glob(jsonl_dir/**/*.jsonl) returns for an absolute jsonl_dir.
    index = {r["jsonl_abs"]: r["frames"] for r in results}
    # remove stale generated episodes outside the current selection
    removed = []
    if not args.keep_stale:
        keep_jsonl = set(index)
        for p in (out / "jsonl").rglob("episode_*.jsonl"):
            if str(p.resolve()) not in keep_jsonl:
                p.unlink()
                removed.append(str(p))
        keep_img = {r["image_dir_rel"] for r in results}
        for p in (out / "images").glob("*/episode_*"):
            if p.is_dir() and str(p.relative_to(out)) not in keep_img:
                shutil.rmtree(p)
                removed.append(str(p))
        if base_col:
            keep_img_base = {r["image_base_dir_rel"] for r in results}
            for p in (out / "images_base").glob("*/episode_*"):
                if p.is_dir() and str(p.relative_to(out)) not in keep_img_base:
                    shutil.rmtree(p)
                    removed.append(str(p))
        for p in (out / "jsonl").glob("*"):
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()
        for p in (out / "images").glob("*"):
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()
        if base_col:
            for p in (out / "images_base").glob("*"):
                if p.is_dir() and not any(p.iterdir()):
                    p.rmdir()
    with open(out / "jsonl" / "index_cache.json", "w") as f:
        json.dump({"data": index}, f, indent=2)

    with open(out / "episodes.jsonl", "w") as f:
        for r in results:
            row = {
                "task": r["task"], "task_dir": r["task_dir"], "task_index": r["task_index"],
                "episode_index": r["episode_index"], "frames": r["frames"],
                "jsonl": r["jsonl_rel"], "image_dir": r["image_dir_rel"], "split": "train",
            }
            if base_col:
                row["image_base_dir"] = r["image_base_dir_rel"]
            f.write(json.dumps(row) + "\n")

    total_frames = sum(r["frames"] for r in results)
    img_bytes = sum(r["image_bytes"] for r in results)
    img_base_bytes = sum(r["image_base_bytes"] for r in results)
    jsonl_bytes = sum(r["jsonl_bytes"] for r in results)
    task_rows = []
    for t in selected_tasks:
        rs = [r for r in results if r["task"] == t]
        if not rs:
            continue
        task_rows.append({
            "task": t, "task_dir": task_dirname(t), "task_index": index_by_task[t],
            "task_name": f"ambench_{task_dirname(t)}",
            "episodes": len(rs), "frames": sum(r["frames"] for r in rs),
        })
    manifest = {
        "format": "opendm_jsonl_v1",
        "source_root": str(src),
        "source_codebase_version": info.get("codebase_version"),
        "source_robot_type": info.get("robot_type"),
        "fps": fps,
        "image_keys": ["images_1", "images_2"] if base_col else ["images_1"],
        "image_prompts": ["Wrist", "Base"] if base_col else ["Wrist"],
        "image_format": args.image_format,
        "image_hw": expect_hw,
        "jsonl_dir": str(out / "jsonl"),
        # two-camera: urls are "images/<Task>/..." and "images_base/<Task>/..." relative to the dataset root
        "image_dir": str(out) if base_col else str(out / "images"),
        **({"image_source_columns": {"images_1": "ee_image", "images_2": base_col},
            "image_wrist_dir": str(out / "images"),
            "image_base_dir": str(out / "images_base")} if base_col else {}),
        "state_dim": 7,
        "action_dim": 7,
        "state_desc": STATE_DESC,
        "state_names": STATE_NAMES,
        "action_names": STATE_NAMES,
        "action_semantics": "absolute end-effector target pose (world frame), same layout as state",
        "rotation": "axis-angle (rotvec) from quaternion wxyz with w>=0, |rotvec|<=pi",
        "gripper_state": "state[6] = gripper_width in metres, >= 0 (unchanged from source)",
        "gripper_action": "action[6] = raw actions[7] gripper COMMAND in [-1, 1]: +1 = open, -1 = close "
                          "(NOT metres; unchanged from source). state and action gripper dims therefore "
                          "have different units; OpenDM normalizes them separately and keeps the gripper "
                          "absolute in RELATIVE action mode.",
        "gripper_state_range": [min(r["gripper_state_range"][0] for r in results),
                                max(r["gripper_state_range"][1] for r in results)],
        "gripper_action_range": [min(r["gripper_action_range"][0] for r in results),
                                 max(r["gripper_action_range"][1] for r in results)],
        "tasks": task_rows,
        "total_episodes": len(results),
        "total_frames": total_frames,
        "image_bytes": img_bytes,
        **({"image_base_bytes": img_base_bytes} if base_col else {}),
        "jsonl_bytes": jsonl_bytes,
        "max_axis_angle_state": max(r["max_axis_angle_state"] for r in results),
        "max_axis_angle_action": max(r["max_axis_angle_action"] for r in results),
        "timestamp_max_dev_s": max(r["timestamp_max_dev"] for r in results),
        "selection": {"tasks": args.tasks, "max_episodes_per_task": args.max_episodes},
        "removed_stale": removed,
        "converter": "env/convert_opendm.py",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t_start, 1),
    }
    with open(out / "dataset.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[convert] done: episodes={len(results)} frames={total_frames} "
          f"images={img_bytes/1e9:.2f}GB images_base={img_base_bytes/1e9:.2f}GB jsonl={jsonl_bytes/1e6:.1f}MB "
          f"max|aa| state={manifest['max_axis_angle_state']:.3f} action={manifest['max_axis_angle_action']:.3f} "
          f"ts_dev={manifest['timestamp_max_dev_s']:.4f}s grip_state={manifest['gripper_state_range']} "
          f"grip_action={manifest['gripper_action_range']} removed_stale={len(removed)} "
          f"elapsed={time.time()-t_start:.1f}s", flush=True)
    for row in task_rows:
        print(f"[convert]   {row['task_dir']:<16s} {row['episodes']:4d} eps {row['frames']:7d} frames", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
