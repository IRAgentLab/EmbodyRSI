"""Drop a truncated trailing episode from a LeRobot v3 session: quarantine its parquet, filter meta/episodes, fix info.json."""
import json, os, shutil, sys
import pyarrow.parquet as pq, pyarrow.compute as pc
root, bad_ep = sys.argv[1], int(sys.argv[2])
bad_file = f"{root}/data/chunk-000/file-{bad_ep:03d}.parquet"
q = f"{root}/../quarantine"; os.makedirs(q, exist_ok=True)
if os.path.exists(bad_file):
    shutil.move(bad_file, f"{q}/file-{bad_ep:03d}.parquet"); print("quarantined", bad_file)
if not os.path.exists(f"{root}/meta.bak"):
    shutil.copytree(f"{root}/meta", f"{root}/meta.bak")
ep_path = f"{root}/meta/episodes/chunk-000/file-000.parquet"
t = pq.read_table(ep_path)
keep = t.filter(pc.less(t["episode_index"], bad_ep))
pq.write_table(keep, ep_path)
info = json.load(open(f"{root}/meta/info.json"))
n = keep.num_rows
frames = int(sum(keep["length"].to_pylist())) if "length" in keep.column_names else None
info["total_episodes"] = n
if frames is not None: info["total_frames"] = frames
info["splits"] = {"train": f"0:{n}"}
json.dump(info, open(f"{root}/meta/info.json", "w"), indent=2)
print(f"episodes {t.num_rows} -> {n}, frames -> {frames}; columns={keep.column_names[:8]}")
