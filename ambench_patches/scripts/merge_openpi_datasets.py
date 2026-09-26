"""Merge several single-task openpi/LeRobot v2.1 exports into one multi-task dataset.

    python merge_openpi_datasets.py OUT_DIR SRC1 SRC2 ...

Renumbers episode/task indices, concatenates meta jsonl files, rebuilds info.json.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1])
srcs = [Path(p) for p in sys.argv[2:]]
if out.exists():
    shutil.rmtree(out)
(out / "data" / "chunk-000").mkdir(parents=True)
(out / "meta").mkdir(parents=True)

tasks, task_idx, episodes, ep_stats = [], {}, [], []
ep_i = frame_i = 0
base_info = None
for src in srcs:
    info = json.loads((src / "meta" / "info.json").read_text())
    base_info = base_info or info
    assert info["fps"] == base_info["fps"], f"fps mismatch in {src}"
    src_tasks = [json.loads(l) for l in (src / "meta" / "tasks.jsonl").read_text().splitlines() if l.strip()]
    remap = {}
    for t in src_tasks:
        if t["task"] not in task_idx:
            task_idx[t["task"]] = len(task_idx)
            tasks.append({"task_index": task_idx[t["task"]], "task": t["task"]})
        remap[t["task_index"]] = task_idx[t["task"]]
    src_eps = [json.loads(l) for l in (src / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    stats_p = src / "meta" / "episodes_stats.jsonl"
    src_stats = [json.loads(l) for l in stats_p.read_text().splitlines() if l.strip()] if stats_p.exists() else []
    for k, ep in enumerate(src_eps):
        idx = ep["episode_index"]
        f = src / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
        if not f.exists():
            print("  missing", f)
            continue
        d = pd.read_parquet(f)
        d["episode_index"] = ep_i
        d["task_index"] = d["task_index"].map(lambda x: remap.get(int(x), 0))
        d["index"] = range(frame_i, frame_i + len(d))
        (out / "data" / f"chunk-{ep_i // 1000:03d}").mkdir(exist_ok=True); d.to_parquet(out / "data" / f"chunk-{ep_i // 1000:03d}" / f"episode_{ep_i:06d}.parquet", index=False)
        episodes.append({"episode_index": ep_i, "tasks": ep["tasks"], "length": ep["length"]})
        if k < len(src_stats):
            st = dict(src_stats[k])
            st["episode_index"] = ep_i
            ep_stats.append(st)
        frame_i += len(d)
        ep_i += 1
    print(f"{src.name}: +{len(src_eps)} eps -> total {ep_i}")

for name, rows in [("tasks.jsonl", tasks), ("episodes.jsonl", episodes), ("episodes_stats.jsonl", ep_stats)]:
    (out / "meta" / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
info = dict(base_info)
info.update(total_episodes=ep_i, total_frames=frame_i, total_tasks=len(tasks), total_chunks=(ep_i + 999) // 1000,
            splits={"train": f"0:{ep_i}"})
(out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
print(f"merged: {ep_i} episodes, {frame_i} frames, {len(tasks)} tasks -> {out}")
