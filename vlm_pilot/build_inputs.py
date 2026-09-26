"""Build model inputs from viewer rollouts: per demo, tile scene|ee|base horizontally per timestep,
write an mp4 (for video-capable models) and a small set of keyframe PNG/JPGs (for image-only models)."""
import glob, json, os, subprocess, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = f"{ROOT}/data"
OUT = f"{ROOT}/inputs"
CASES = {  # case_id: (entry dir relative to DATA without ambench_rollouts/data prefix, split, label)
    "frame_fail":  ("_trash", "FrameAssemblyFAHexaAbsPID_expert3", "fail"),
    "toss_fail":   ("_trash", "TossBallFAHexaAbsPID_expert3", "fail"),
    "wipe_fail":   ("_trash", "WipeWindowFAHexaAbsPID_expert3", "fail"),
    "frame_ok":    ("", "FrameAssemblyFAHexaAbsPID_fix", "ok"),
    "toss_ok":     ("", "TossBallFAHexaAbsPID_ballistic", "ok"),
    "wipe_ok":     ("", "WipeWindowFAHexaAbsPID_servo3", "ok"),
}
N_KEY = int(os.environ.get("N_KEY", "12"))
FPS = int(os.environ.get("FPS", "4"))
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-800:]); raise SystemExit(1)
manifest = []
for cid, (pre, entry, label) in CASES.items():
    base = f"{DATA}/{pre + '/' if pre else ''}ambench_rollouts/{entry}"
    jdir = f"{DATA}/{pre + '/' if pre else ('viewer/')}ambench_data/{entry}"
    picked = 0
    for demo in sorted(glob.glob(f"{base}/demo_*")):
        if picked >= 2: break
        dn = os.path.basename(demo)
        meta = json.load(open(f"{jdir}/{dn}.json")).get("meta", {})
        if bool(meta.get("success")) != (label == "ok"): continue
        picked += 1
        steps = sorted({int(f.rsplit("_", 1)[1].split(".")[0]) for f in glob.glob(f"{demo}/frames/scene_*.jpg")})
        od = f"{OUT}/{cid}/{dn}"; os.makedirs(f"{od}/tiles", exist_ok=True)
        for i, s in enumerate(steps):
            t = f"{od}/tiles/{i:04d}.jpg"
            if not os.path.exists(t):
                run(["ffmpeg", "-loglevel", "error", "-y", "-i", f"{demo}/frames/scene_{s:06d}.jpg", "-i", f"{demo}/frames/ee_{s:06d}.jpg",
                     "-i", f"{demo}/frames/base_{s:06d}.jpg", "-filter_complex",
                     "[0]scale=-2:360[a];[1]scale=-2:360[b];[2]scale=-2:360[c];[a][b][c]hstack=3,scale=1920:-2",
                     "-q:v", "3", t])
        run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", f"{od}/tiles/%04d.jpg",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=1920:-2", f"{od}/video.mp4"])
        idx = [round(k * (len(steps) - 1) / (N_KEY - 1)) for k in range(N_KEY)]
        keys = [f"{od}/tiles/{i:04d}.jpg" for i in idx]
        key_t = [round(steps[i] / 20, 1) for i in idx]
        manifest.append({"case": cid, "demo": dn, "label": label, "success": meta.get("success"),
                         "termination": meta.get("termination_reason"), "n_frames": len(steps),
                         "duration_s": round(steps[-1] / 20, 1), "video": f"{od}/video.mp4", "keyframes": keys, "keyframe_t": key_t})
        print(cid, dn, label, "success=", meta.get("success"), "frames=", len(steps), "dur=", round(steps[-1] / 20, 1), "s")
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), ensure_ascii=False, indent=1)
