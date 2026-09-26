#!/usr/bin/env python
"""
AM-Bench LeRobot v2.1 (PNG bytes embedded in parquet) -> Being-H0.5 readable
LeRobot v2.1 (mp4 videos + observation.state / action columns).

Source layout (per episode parquet):
    ee_image      dict{bytes: PNG, path: None}   RGB 384x384
    actions       float32[8]  absolute EE target pose [x y z qw qx qy qz gripper_cmd]
    ee_pos        float32[3]
    ee_quat       float32[4]  (w x y z)
    gripper_width float32     (meters, scalar or [1])
    timestamp / frame_index / episode_index / index / task_index

Target layout (LeRobot v2.1, what Being-H05 BeingH/dataset/datasets/vla_dataset.py reads):
    meta/info.json            with video_path template + chunks_size + video feature "info"
    meta/episodes.jsonl       {episode_index, tasks, length, ...}
    meta/tasks.jsonl          {task_index, task}
    meta/episodes_stats.jsonl LeRobot v2.1 per-episode stats (min/max/mean/std/count)
    meta/stats.json           Being-H global stats per column (mean/std/min/max/q01/q99)
    data/chunk-XXX/episode_XXXXXX.parquet
        observation.state float32[7] = [x y z ax ay az gripper_width]   (axis-angle, |aa| <= pi)
        action            float32[7] = relative (default) or absolute pose + gripper cmd
        timestamp frame_index episode_index index task_index
    videos/chunk-XXX/<video_key>/episode_XXXXXX.mp4   h264 yuv420p, fps=source fps
    (--extra-image <col>:<video_key>, repeatable, adds one more mp4 stream per episode
     from another image column, e.g. base_image:observation.images.base_camera; it gets
     the same video feature entry in info.json and its own image stats in episodes_stats.jsonl)

Relative action (default, mirrors Being-H `is_relative` code path):
    d_pos = R(q_s)^T (p_a - p_s)          (expressed in current EE frame)
    d_rot = as_rotvec( q_s^{-1} (x) q_a ) (axis-angle, |d_rot| <= pi)
    gripper = actions[7] as-is, NOT differenced (NOTE: in AM-Bench this is a +-1 command,
              +1 = open / -1 = close, ramping continuously in between; NOT meters)

Heavy work (parquet parse / PNG decode / video encode) -> run as an sbatch CPU job.
Single-process, serial subprocess encoding (LeRobot's multi-process encoder hit Errno 11 here).
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CANON_ALIASES = {
    "PressButton": "press the button",
    "PegInHole": "insert the peg into the hole",
    "PushSlider": "push the slider to the right",
    "OpenDoor": "open the door",
    "LemonHarvesting": "pick the yellow lemon and place it in the bin",
    "PullLever": "pull the lever upwards",
    "RotateValve": "rotate the valve clockwise",
    "NDT": "contact the inspection point and hold",
    "CabinetPickPlace": "open the cabinet sliding door, pick up the red can, and place it on top of the cabinet",
}

STATE_NAMES = ["x", "y", "z", "ax", "ay", "az", "gripper"]
ACTION_NAMES = ["x", "y", "z", "ax", "ay", "az", "gripper"]

# N0-TWAM（neoteai/N0-TWAM）的 20 维双臂 EE схема：[left xyz, rot6d, grip | right xyz, rot6d, grip]，
# 单臂只填前 10 维、后 10 维置 0（POST_TRAINING.md §1）。action 必须是绝对位姿（absee 配方）。
N0_NAMES = ([f"l_{n}" for n in ("x", "y", "z", "r6_0", "r6_1", "r6_2", "r6_3", "r6_4", "r6_5", "grip")]
            + [f"r_{n}" for n in ("x", "y", "z", "r6_0", "r6_1", "r6_2", "r6_3", "r6_4", "r6_5", "grip")])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- math
def quat_wxyz_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    q = q / n
    # canonical hemisphere: w >= 0
    neg = q[..., 0] < 0
    q[neg] = -q[neg]
    return q


def quat_wxyz_to_rotvec(q):
    """Quaternion (w,x,y,z), w>=0 -> axis-angle with |theta| <= pi."""
    q = quat_wxyz_normalize(q)
    w = np.clip(q[..., 0], -1.0, 1.0)
    v = q[..., 1:4]
    sin_half = np.linalg.norm(v, axis=-1)
    theta = 2.0 * np.arctan2(sin_half, w)  # in [0, pi] since w >= 0
    axis = np.zeros_like(v)
    ok = sin_half > 1e-12
    axis[ok] = v[ok] / sin_half[ok][:, None]
    return axis * theta[:, None]


def quat_wxyz_to_matrix(q):
    q = quat_wxyz_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def quat_wxyz_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_wxyz_conj(q):
    q = np.array(q, dtype=np.float64, copy=True)
    q[..., 1:] *= -1
    return q


def compute_state_action(ee_pos, ee_quat, gripper_width, actions, mode):
    """Return (state[N,7], action[N,7]) float32."""
    p_s = np.asarray(ee_pos, dtype=np.float64)
    q_s = quat_wxyz_normalize(ee_quat)
    g_s = np.asarray(gripper_width, dtype=np.float64).reshape(-1, 1)
    a = np.asarray(actions, dtype=np.float64)
    p_a = a[:, 0:3]
    q_a = quat_wxyz_normalize(a[:, 3:7])
    g_a = a[:, 7:8]

    state = np.concatenate([p_s, quat_wxyz_to_rotvec(q_s), g_s], axis=1)
    if mode == "absolute":
        action = np.concatenate([p_a, quat_wxyz_to_rotvec(q_a), g_a], axis=1)
    elif mode == "relative":
        R_s = quat_wxyz_to_matrix(q_s)                       # [N,3,3]
        d_pos = np.einsum("nji,nj->ni", R_s, p_a - p_s)      # R^T (p_a - p_s)
        q_rel = quat_wxyz_mul(quat_wxyz_conj(q_s), q_a)      # q_s^-1 (x) q_a
        d_rot = quat_wxyz_to_rotvec(q_rel)
        action = np.concatenate([d_pos, d_rot, g_a], axis=1)
    else:
        raise ValueError(mode)
    return state.astype(np.float32), action.astype(np.float32)


def quat_wxyz_to_rot6d(q):
    """Zhou et al. 6D 旋转表示：旋转矩阵的前两列（列优先拼接）。"""
    R = quat_wxyz_to_matrix(q)                      # [N,3,3]
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1)


def compute_state_action_n0twam(ee_pos, ee_quat, gripper_width, actions):
    """N0-TWAM 20 维：state = [xyz, rot6d, gripper_width(m), 0*10]，
    action = 绝对目标 [xyz, rot6d, grip∈[0,1], 0*10]（±1 指令映射为 (c+1)/2，1=开）。"""
    p_s = np.asarray(ee_pos, dtype=np.float64)
    q_s = quat_wxyz_normalize(ee_quat)
    g_s = np.asarray(gripper_width, dtype=np.float64).reshape(-1, 1)
    a = np.asarray(actions, dtype=np.float64)
    p_a, q_a = a[:, 0:3], quat_wxyz_normalize(a[:, 3:7])
    g_a = np.clip((a[:, 7:8] + 1.0) / 2.0, 0.0, 1.0)
    n = len(p_s)
    zeros = np.zeros((n, 10))
    state = np.concatenate([p_s, quat_wxyz_to_rot6d(q_s), g_s, zeros], axis=1)
    action = np.concatenate([p_a, quat_wxyz_to_rot6d(q_a), g_a, zeros], axis=1)
    return state.astype(np.float32), action.astype(np.float32)


# --------------------------------------------------------------------------- video
class Encoder:
    """Serial video encoder. Priority: system ffmpeg > imageio_ffmpeg binary > PyAV > cv2."""

    def __init__(self, fps, crf=18, gop=1, prefer=None):
        self.fps = fps
        self.crf = crf
        self.gop = gop
        self.backend = None
        self.ffmpeg = None
        self.vcodec = None
        order = [prefer] if prefer else []
        order += ["ffmpeg", "imageio_ffmpeg", "pyav", "cv2"]
        for b in order:
            if self._try(b):
                break
        if self.backend is None:
            raise RuntimeError("no usable video encoder (pip install imageio-ffmpeg)")

    def _try(self, b):
        if b == "ffmpeg":
            exe = shutil.which("ffmpeg")
            if exe:
                return self._setup_ffmpeg(exe, "system ffmpeg")
        elif b == "imageio_ffmpeg":
            try:
                import imageio_ffmpeg
                exe = imageio_ffmpeg.get_ffmpeg_exe()
                return self._setup_ffmpeg(exe, "imageio_ffmpeg")
            except Exception as e:
                log(f"imageio_ffmpeg unavailable: {e!r}")
        elif b == "pyav":
            try:
                import av  # noqa
                self.backend = "pyav"
                self.vcodec = "libx264"
                return True
            except Exception as e:
                log(f"pyav unavailable: {e!r}")
        elif b == "cv2":
            try:
                import cv2  # noqa
                self.backend = "cv2"
                self.vcodec = "mp4v"
                return True
            except Exception as e:
                log(f"cv2 unavailable: {e!r}")
        return False

    def _setup_ffmpeg(self, exe, label):
        try:
            enc = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=60).stdout
        except Exception as e:
            log(f"{label} at {exe} failed: {e!r}")
            return False
        for c in ("libx264", "libopenh264", "mpeg4"):
            if f" {c} " in enc:
                self.vcodec = c
                break
        if self.vcodec is None:
            log(f"{label}: no h264/mpeg4 encoder found")
            return False
        self.backend = "ffmpeg"
        self.ffmpeg = exe
        self.label = label
        return True

    @property
    def codec_name(self):
        return {"libx264": "h264", "libopenh264": "h264", "mpeg4": "mpeg4", "mp4v": "mpeg4"}[self.vcodec]

    def describe(self):
        return f"backend={self.backend} exe={self.ffmpeg} vcodec={self.vcodec} crf={self.crf} gop={self.gop} fps={self.fps}"

    def encode(self, frames_iter, n_frames, h, w, out_path):
        """frames_iter yields HxWx3 uint8 RGB arrays. Writes atomically to out_path."""
        tmp = out_path.with_name(out_path.name + ".tmp.mp4")
        if tmp.exists():
            tmp.unlink()
        if self.backend == "ffmpeg":
            cmd = [self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-",
                   "-an", "-fps_mode", "passthrough", "-c:v", self.vcodec, "-pix_fmt", "yuv420p", "-g", str(self.gop)]
            if self.vcodec == "libx264":
                cmd += ["-crf", str(self.crf), "-preset", "medium"]
            elif self.vcodec == "mpeg4":
                cmd += ["-q:v", "2"]
            cmd += ["-movflags", "+faststart", str(tmp)]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                for fr in frames_iter:
                    proc.stdin.write(np.ascontiguousarray(fr).tobytes())
                proc.stdin.close()
            except BrokenPipeError:
                pass
            err = proc.stderr.read().decode(errors="replace")
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg rc={rc}: {err[-2000:]}")
        elif self.backend == "pyav":
            import av
            with av.open(str(tmp), mode="w") as c:
                st = c.add_stream(self.vcodec, rate=self.fps)
                st.width, st.height, st.pix_fmt = w, h, "yuv420p"
                st.options = {"crf": str(self.crf), "g": str(self.gop)}
                for fr in frames_iter:
                    vf = av.VideoFrame.from_ndarray(fr, format="rgb24")
                    for pkt in st.encode(vf):
                        c.mux(pkt)
                for pkt in st.encode():
                    c.mux(pkt)
        elif self.backend == "cv2":
            import cv2
            vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            for fr in frames_iter:
                vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            vw.release()
        os.replace(tmp, out_path)


# --------------------------------------------------------------------------- stats helpers
def lerobot_stats_vec(arr):
    """LeRobot v2.1 per-episode stats for a [N,D] (or [N]) numeric array."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(),
            "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [int(a.shape[0])]}


class ImageStatsAcc:
    """Per-channel stats over decoded frames, LeRobot style ([0,1] range, shape (3,1,1))."""

    def __init__(self):
        self.n = 0
        self.s = np.zeros(3)
        self.ss = np.zeros(3)
        self.mn = np.full(3, np.inf)
        self.mx = np.full(3, -np.inf)

    def add(self, fr):
        f = fr.reshape(-1, 3).astype(np.float64) / 255.0
        self.n += f.shape[0]
        self.s += f.sum(0)
        self.ss += (f * f).sum(0)
        self.mn = np.minimum(self.mn, f.min(0))
        self.mx = np.maximum(self.mx, f.max(0))

    def result(self, count):
        mean = self.s / max(self.n, 1)
        var = np.maximum(self.ss / max(self.n, 1) - mean ** 2, 0)
        r = lambda v: [[[float(x)]] for x in v]
        return {"min": r(self.mn), "max": r(self.mx), "mean": r(mean), "std": r(np.sqrt(var)), "count": [int(count)]}


def beingh_stats(df, skip=("episode_index", "index")):
    """Global stats in the format of BeingH/dataset/parquet_utils.calculate_dataset_statistics."""
    out = {}
    for col in df.columns:
        if col in skip:
            continue
        first = df[col].dropna().iloc[0]
        if isinstance(first, (np.ndarray, list)):
            x = np.stack(df[col].dropna().to_numpy()).astype(np.float64)
        elif np.isscalar(first):
            x = df[col].dropna().to_numpy(dtype=np.float64).reshape(-1, 1)
        else:
            continue
        out[col] = {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
                    "min": x.min(0).tolist(), "max": x.max(0).tolist(),
                    "q01": np.quantile(x, 0.01, axis=0).tolist(), "q99": np.quantile(x, 0.99, axis=0).tolist()}
    return out


# --------------------------------------------------------------------------- io helpers
def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    tmp = Path(str(p) + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, p)


def write_json(p, obj):
    tmp = Path(str(p) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, p)


def decode_png(cell):
    if isinstance(cell, dict):
        b = cell["bytes"]
    elif isinstance(cell, (bytes, bytearray)):
        b = cell
    else:
        raise TypeError(f"unexpected image cell type {type(cell)}")
    im = Image.open(io.BytesIO(b))
    if im.mode != "RGB":
        im = im.convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def resolve_tasks(requested, tasks):
    """requested: list of strings (task text or CamelCase alias). tasks: {index: text}."""
    if not requested:
        return sorted(tasks.keys())
    by_text = {t.lower().strip(): i for i, t in tasks.items()}
    sel = []
    for r in requested:
        key = r.strip()
        text = CANON_ALIASES.get(key, key).lower()
        if text not in by_text:
            raise SystemExit(f"task {r!r} not found in tasks.jsonl; available: {list(tasks.values())} "
                             f"or aliases {list(CANON_ALIASES)}")
        sel.append(by_text[text])
    return sorted(set(sel))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="LeRobot v2.1 root (with meta/, data/)")
    ap.add_argument("--out", required=True, help="output root")
    ap.add_argument("--tasks", nargs="*", default=None, help="task strings or CamelCase aliases; default all")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--action-mode", choices=["relative", "absolute"], default="relative")
    ap.add_argument("--target", choices=["beingh", "n0twam"], default="beingh",
                    help="beingh: 7 维轴角、相对/绝对由 --action-mode 决定；"
                         "n0twam: 20 维 [xyz, rot6d, grip, 0*10] 绝对位姿（忽略 --action-mode）")
    ap.add_argument("--video-size", type=int, default=None,
                    help="把帧缩放到 N×N 再编码（N0-TWAM 的 VAE 用 256）；默认不缩放")
    ap.add_argument("--tactile-zero", default=None, metavar="KEY",
                    help="额外写一路全黑触觉视频（如 observation.images.tactile_a），给无触觉本体占位")
    ap.add_argument("--video-key", default="observation.images.ee_camera")
    ap.add_argument("--image-col", default="ee_image")
    ap.add_argument("--extra-image", action="append", default=None, metavar="COL:VIDEO_KEY",
                    help="additional image column -> video key, repeatable, e.g. "
                         "base_image:observation.images.base_camera (default none = single camera)")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--gop", type=int, default=1,
                    help="keyframe interval; 1 = all-intra (Being-H torchvision_av backend returns every frame from the "
                         "previous keyframe up to the requested timestamp, so keep this small)")
    ap.add_argument("--encoder", choices=["ffmpeg", "imageio_ffmpeg", "pyav", "cv2"], default=None,
                    help="force a backend (default priority: ffmpeg > imageio_ffmpeg > pyav > cv2)")
    ap.add_argument("--timestamp-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--chunks-size", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true", help="re-encode episodes even if outputs exist")
    args = ap.parse_args()
    extra_images = []   # [(source column, video key)]
    for spec in (args.extra_image or []):
        if ":" not in spec:
            raise SystemExit(f"--extra-image expects COL:VIDEO_KEY, got {spec!r}")
        col, key = spec.split(":", 1)
        col, key = col.strip(), key.strip()
        if not col or not key or col == args.image_col or key == args.video_key or key == args.tactile_zero:
            raise SystemExit(f"--extra-image {spec!r}: empty or clashes with --image-col/--video-key/--tactile-zero")
        if any(k == key for _, k in extra_images):
            raise SystemExit(f"--extra-image: duplicate video key {key!r}")
        extra_images.append((col, key))

    t0 = time.time()
    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    info = json.load(open(src / "meta/info.json"))
    fps = float(info["fps"])
    src_tasks = {r["task_index"]: r["task"] for r in read_jsonl(src / "meta/tasks.jsonl")}
    src_eps = read_jsonl(src / "meta/episodes.jsonl")
    src_chunks = int(info.get("chunks_size", 1000))
    data_tpl = info["data_path"]

    sel_task_idx = resolve_tasks(args.tasks, src_tasks)
    new_task_index = {old: new for new, old in enumerate(sel_task_idx)}
    sel_task_text = {src_tasks[old] for old in sel_task_idx}
    episodes = [e for e in src_eps if any(t in sel_task_text for t in e.get("tasks", []))]
    episodes.sort(key=lambda e: e["episode_index"])
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise SystemExit("no episodes selected")
    log(f"src={src} fps={fps} tasks={[src_tasks[i] for i in sel_task_idx]} episodes={len(episodes)} mode={args.action_mode}"
        + (f" extra_images={extra_images}" if extra_images else ""))
    for col, _ in extra_images:
        if col not in info.get("features", {}):
            raise SystemExit(f"--extra-image column {col!r} not in source meta/info.json features")

    enc = Encoder(fps, crf=args.crf, gop=args.gop, prefer=args.encoder)
    log(f"encoder: {enc.describe()}")

    vk = args.video_key
    (out / "meta").mkdir(parents=True, exist_ok=True)
    video_tpl = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    out_data_tpl = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"

    ep_rows, stat_rows, ep_map = [], [], []
    H = W = None
    global_index = 0
    n_done_new = 0
    for k, e in enumerate(episodes):
        old_ep = e["episode_index"]
        new_ep = k
        chunk = new_ep // args.chunks_size
        pq_in = src / data_tpl.format(episode_chunk=old_ep // src_chunks, episode_index=old_ep)
        pq_out = out / out_data_tpl.format(episode_chunk=chunk, episode_index=new_ep)
        mp4_out = out / video_tpl.format(episode_chunk=chunk, video_key=vk, episode_index=new_ep)
        extra_outs = [(col, key, out / video_tpl.format(episode_chunk=chunk, video_key=key, episode_index=new_ep))
                      for col, key in extra_images]
        stats_out = out / "meta" / "_episode_stats" / f"episode_{new_ep:06d}.json"
        for p in (pq_out, mp4_out, stats_out, *[x[2] for x in extra_outs]):
            p.parent.mkdir(parents=True, exist_ok=True)

        done = (not args.overwrite and pq_out.exists() and mp4_out.exists() and stats_out.exists()
                and pq_out.stat().st_size > 0 and mp4_out.stat().st_size > 0
                and all(x[2].exists() and x[2].stat().st_size > 0 for x in extra_outs))
        if done and extra_outs:
            # an episode encoded by an earlier single-camera run has no stats for the extra streams -> redo it
            _st_prev = json.load(open(stats_out))
            done = all(key in _st_prev.get("stats", {}) for _, key, _ in extra_outs)
        if done:
            st = json.load(open(stats_out))
            n = st["length"]
            H, W = st["height"], st["width"]
            # patch index/episode columns if global numbering shifted (e.g. different selection)
            if st.get("index_start") != global_index or st.get("task_index") != new_task_index[st["src_task_index"]]:
                df = pd.read_parquet(pq_out)
                df["index"] = np.arange(global_index, global_index + n, dtype=np.int64)
                df["task_index"] = np.int64(new_task_index[st["src_task_index"]])
                df.to_parquet(pq_out, index=False)
                st["index_start"] = global_index
                st["task_index"] = new_task_index[st["src_task_index"]]
                st["stats"]["index"] = lerobot_stats_vec(df["index"].to_numpy())
                st["stats"]["task_index"] = lerobot_stats_vec(df["task_index"].to_numpy())
                write_json(stats_out, st)
            log(f"[{k+1}/{len(episodes)}] episode {old_ep}->{new_ep}: exists, skip ({n} frames)")
        else:
            te = time.time()
            df = pd.read_parquet(pq_in)
            n = len(df)
            src_task_index = int(df["task_index"].iloc[0])
            if src_task_index not in new_task_index:
                raise SystemExit(f"episode {old_ep}: task_index {src_task_index} not in selection")
            ee_pos = np.stack(df["ee_pos"].to_numpy())
            ee_quat = np.stack(df["ee_quat"].to_numpy())
            gw = df["gripper_width"].to_numpy()
            if gw.dtype == object:
                gw = np.stack(gw).reshape(n, -1)[:, 0]
            actions = np.stack(df["actions"].to_numpy())
            if args.target == "n0twam":
                state, action = compute_state_action_n0twam(ee_pos, ee_quat, gw, actions)
            else:
                state, action = compute_state_action(ee_pos, ee_quat, gw, actions, args.action_mode)

            frame_index = df["frame_index"].to_numpy().astype(np.int64)
            if not np.array_equal(frame_index, np.arange(n)):
                raise SystemExit(f"episode {old_ep}: frame_index not contiguous 0..{n-1}")
            ts = (frame_index / fps).astype(np.float32 if args.timestamp_dtype == "float32" else np.float64)

            img_acc = ImageStatsAcc()
            def _prep(fr):
                if args.video_size and fr.shape[:2] != (args.video_size, args.video_size):
                    fr = np.asarray(Image.fromarray(fr).resize((args.video_size, args.video_size), Image.BILINEAR))
                return fr

            first = _prep(decode_png(df[args.image_col].iloc[0]))
            H, W = first.shape[:2]

            def frames():
                for i in range(n):
                    fr = first if i == 0 else _prep(decode_png(df[args.image_col].iloc[i]))
                    if fr.shape[:2] != (H, W):
                        raise SystemExit(f"episode {old_ep} frame {i}: shape {fr.shape} != {(H, W)}")
                    img_acc.add(fr)
                    yield fr

            enc.encode(frames(), n, H, W, mp4_out)
            extra_stats = {}
            for col, key, ex_out in extra_outs:
                ex_acc = ImageStatsAcc()

                def ex_frames(_col=col, _acc=ex_acc):
                    for i in range(n):
                        fr = _prep(decode_png(df[_col].iloc[i]))
                        if fr.shape[:2] != (H, W):
                            raise SystemExit(f"episode {old_ep} {_col} frame {i}: shape {fr.shape} != {(H, W)}")
                        _acc.add(fr)
                        yield fr

                enc.encode(ex_frames(), n, H, W, ex_out)
                extra_stats[key] = ex_acc.result(n)
            if args.tactile_zero:
                tac_out = out / video_tpl.format(episode_chunk=new_ep // args.chunks_size,
                                                 video_key=args.tactile_zero, episode_index=new_ep)
                tac_out.parent.mkdir(parents=True, exist_ok=True)
                black = np.zeros((H, W, 3), dtype=np.uint8)
                enc.encode((black for _ in range(n)), n, H, W, tac_out)

            odf = pd.DataFrame({
                "observation.state": list(state),
                "action": list(action),
                "timestamp": ts,
                "frame_index": frame_index,
                "episode_index": np.full(n, new_ep, dtype=np.int64),
                "index": np.arange(global_index, global_index + n, dtype=np.int64),
                "task_index": np.full(n, new_task_index[src_task_index], dtype=np.int64),
            })
            tmp = pq_out.with_name(pq_out.name + ".tmp")
            odf.to_parquet(tmp, index=False)
            os.replace(tmp, pq_out)

            st = {
                "length": n, "height": H, "width": W, "index_start": global_index,
                "src_episode_index": old_ep, "src_task_index": src_task_index,
                "task_index": new_task_index[src_task_index],
                "stats": {
                    vk: img_acc.result(n),
                    **extra_stats,
                    "observation.state": lerobot_stats_vec(state),
                    "action": lerobot_stats_vec(action),
                    "timestamp": lerobot_stats_vec(ts),
                    "frame_index": lerobot_stats_vec(frame_index),
                    "episode_index": lerobot_stats_vec(odf["episode_index"].to_numpy()),
                    "index": lerobot_stats_vec(odf["index"].to_numpy()),
                    "task_index": lerobot_stats_vec(odf["task_index"].to_numpy()),
                },
            }
            write_json(stats_out, st)
            n_done_new += 1
            el = time.time() - t0
            eta = el / (k + 1) * (len(episodes) - k - 1)
            log(f"[{k+1}/{len(episodes)}] episode {old_ep}->{new_ep}: {n} frames, {time.time()-te:.1f}s, "
                f"mp4 {mp4_out.stat().st_size/1e6:.1f} MB"
                + "".join(f", {key.split('.')[-1]} {ex_out.stat().st_size/1e6:.1f} MB" for _, key, ex_out in extra_outs)
                + f", elapsed {el/60:.1f} min, eta {eta/60:.1f} min")

        ep_rows.append({"episode_index": new_ep, "tasks": [src_tasks[st["src_task_index"]]], "length": n,
                        "source_episode_index": st["src_episode_index"]})
        stat_rows.append({"episode_index": new_ep, "stats": st["stats"]})
        ep_map.append({"episode_index": new_ep, "source_episode_index": st["src_episode_index"],
                       "length": n, "task_index": st["task_index"]})
        global_index += n

    total_frames = global_index
    total_eps = len(episodes)
    total_chunks = (total_eps - 1) // args.chunks_size + 1

    # ----- meta
    write_jsonl(out / "meta/tasks.jsonl", [{"task_index": new_task_index[o], "task": src_tasks[o]} for o in sel_task_idx])
    write_jsonl(out / "meta/episodes.jsonl", ep_rows)
    write_jsonl(out / "meta/episodes_stats.jsonl", stat_rows)

    features = {
        vk: {"dtype": "video", "shape": [H, W, 3], "names": ["height", "width", "channel"],
             "info": {"video.fps": fps, "video.height": H, "video.width": W, "video.channels": 3,
                      "video.codec": enc.codec_name, "video.pix_fmt": "yuv420p",
                      "video.is_depth_map": False, "has_audio": False}},
        "observation.state": {"dtype": "float32", "shape": [20 if args.target == "n0twam" else 7],
                              "names": N0_NAMES if args.target == "n0twam" else STATE_NAMES},
        "action": {"dtype": "float32", "shape": [20 if args.target == "n0twam" else 7],
                   "names": N0_NAMES if args.target == "n0twam" else ACTION_NAMES},
        "timestamp": {"dtype": args.timestamp_dtype, "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    if args.tactile_zero:
        features[args.tactile_zero] = dict(features[vk])
    for _, key in extra_images:
        features[key] = dict(features[vk])
    # keep the video features grouped in front of the state/action columns (LeRobot convention)
    if extra_images:
        video_feats = {k: v for k, v in features.items() if v["dtype"] == "video"}
        features = {**video_feats, **{k: v for k, v in features.items() if v["dtype"] != "video"}}
    new_info = {
        "codebase_version": "v2.1",
        "robot_type": info.get("robot_type", "am_bench"),
        "total_episodes": total_eps,
        "total_frames": total_frames,
        "total_tasks": len(sel_task_idx),
        "total_videos": total_eps * (1 + len(extra_images)) if extra_images else total_eps,
        "total_chunks": total_chunks,
        "chunks_size": args.chunks_size,
        "fps": fps,
        "splits": {"train": f"0:{total_eps}"},
        "data_path": out_data_tpl,
        "video_path": video_tpl,
        "features": features,
        "ambench": {
            "beingh_export": {
                "source_root": str(src),
                "source_image_column": args.image_col,
                "target": args.target,
                "action_mode": "absolute" if args.target == "n0twam" else args.action_mode,
                "video_size": args.video_size,
                "tactile_zero_key": args.tactile_zero,
                **({"extra_images": [{"source_image_column": c, "video_key": k} for c, k in extra_images]}
                   if extra_images else {}),
                "state_layout": ("[x,y,z, rot6d(6), gripper_width (m), 0*10]" if args.target == "n0twam" else
                                 "[x,y,z, ax,ay,az (axis-angle, |aa|<=pi), gripper_width (m)]"),
                "action_layout": ("[x,y,z (world abs), rot6d(6), grip=(cmd+1)/2 in [0,1] (1=open), 0*10]" if args.target == "n0twam" else
                                  "[dx,dy,dz (in current EE frame), dax,day,daz (axis-angle of q_s^-1 q_a), gripper_cmd (+-1)]"
                                  if args.action_mode == "relative" else
                                  "[x,y,z (world), ax,ay,az (axis-angle), gripper_cmd (+-1)]"),
                "quaternion_source_order": "wxyz",
                "selected_source_task_index": sel_task_idx,
                "task_index_map_src_to_new": {str(k): v for k, v in new_task_index.items()},
                "encoder": enc.describe(),
                "episode_map": ep_map,
            }
        },
    }
    write_json(out / "meta/info.json", new_info)

    # ----- Being-H global stats (meta/stats.json); read back output parquet (no images, cheap)
    log("computing meta/stats.json ...")
    dfs = [pd.read_parquet(out / out_data_tpl.format(episode_chunk=i // args.chunks_size, episode_index=i))
           for i in range(total_eps)]
    all_df = pd.concat(dfs, ignore_index=True)
    write_json(out / "meta/stats.json", beingh_stats(all_df))

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    log(f"DONE episodes={total_eps} frames={total_frames} new_encoded={n_done_new} "
        f"out_size={size/1e9:.2f} GB elapsed={(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
