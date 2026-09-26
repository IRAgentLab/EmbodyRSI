"""把模型评测回放（每个相机一个 mp4）做成与专家案例相同格式的三视角输入，并追加到 inputs/manifest.json。
这些是"真正的策略失败"对照组：环境没有问题，失败原因在策略或观测。"""
import glob, json, os, subprocess
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = f"{ROOT}/data/eval"
OUT = f"{ROOT}/inputs"
CASES = {  # case_id: (job dir, [episode ids])
    "push_policy": ("383328_PushSlider", [0, 1]),
    "peg_policy": ("385030_PegInHole", [0, 1]),
    "pull_policy": ("386370_PullLever", [1, 5]),
}
N_KEY, FPS, STEP = 12, 4, 0.5   # 与专家案例一致：每 0.5 s 仿真时间取一帧
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-800:]); raise SystemExit(1)
def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p], capture_output=True, text=True).stdout)
man = [m for m in json.load(open(f"{OUT}/manifest.json")) if m["case"] not in CASES]
for m in man:  # 给原有案例补上真值标签
    m["truth"] = {"fail": "env", "ok": "ok"}.get(m["label"], m.get("truth"))
for cid, (job, eps) in CASES.items():
    for k, ep in enumerate(eps):
        v = {c: f"{SRC}/{job}/videos/pi-eval-{c}_camera-env0-eps{ep}.mp4" for c in ("scene", "ee", "base")}
        d = min(dur(p) for p in v.values())
        od = f"{OUT}/{cid}/demo_{k:04d}"; os.makedirs(f"{od}/tiles", exist_ok=True)
        run(["ffmpeg", "-loglevel", "error", "-y", "-i", v["scene"], "-i", v["ee"], "-i", v["base"], "-filter_complex",
             f"[0]fps={1/STEP},scale=-2:360[a];[1]fps={1/STEP},scale=-2:360[b];[2]fps={1/STEP},scale=-2:360[c];[a][b][c]hstack=3,scale=1920:-2",
             "-q:v", "3", f"{od}/tiles/%04d.jpg"])
        tiles = sorted(glob.glob(f"{od}/tiles/*.jpg"))
        run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", f"{od}/tiles/%04d.jpg",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=1920:-2", f"{od}/video.mp4"])
        idx = [round(i * (len(tiles) - 1) / (N_KEY - 1)) for i in range(N_KEY)]
        man.append({"case": cid, "demo": f"demo_{k:04d}", "label": "fail", "truth": "policy", "success": False,
                    "source_eval": f"{job}/eps{ep}", "n_frames": len(tiles), "duration_s": round(d, 1),
                    "video": f"{od}/video.mp4", "keyframes": [tiles[i] for i in idx], "keyframe_t": [round(i * STEP, 1) for i in idx]})
        print(cid, k, job, "eps", ep, "frames", len(tiles), "dur", round(d, 1))
json.dump(man, open(f"{OUT}/manifest.json", "w"), ensure_ascii=False, indent=1)
print("manifest:", len(man), "items;", {t: sum(m["truth"] == t for m in man) for t in ("env", "policy", "ok")})
