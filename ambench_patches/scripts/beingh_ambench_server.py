#!/usr/bin/env python3
"""Being-H0.5 (Being-H05-2B) <-> AM-Bench openpi-websocket bridge.

Wire protocol: identical to openpi's websocket_policy_server (msgpack with the
numpy ext hook, first frame = metadata dict, then obs -> action frames), the same
protocol env/cosmos_ambench_server.py and env/lingbot_bridge.py speak, so
env/test_policy_client.py and ambench_learn.policies.pi.eval can talk to it.

Client sends (see $A/env/test_policy_client.py):
    am_bench/ee_image     HxWx3 uint8     -> video.wrist_view   (Being-H resizes to 224)
    am_bench/base_image   HxWx3 uint8     -> unused (the AM-Bench data config is wrist-only)
    am_bench/ee_pos       [3]             -> state.eef_position
    am_bench/ee_quat      [4] wxyz        -> state.eef_rotation (axis-angle, w>=0 canonical, |aa|<=pi)
    am_bench/gripper_width[1] metres      -> state.gripper
    prompt                str             -> language.instruction
Returns {"actions": (HORIZON, 8) float32} = absolute EE [xyz, quat wxyz, gripper +-1].

Being-H predicts a 16-step chunk of *relative* actions in the current EE frame
(exactly what env/convert_beingh.py wrote): d_pos = R_s^T (p_a - p_s),
d_rot = q_s^-1 * q_a. We integrate them forward, chained (each step relative to
the pose reached by the previous step) and pad to HORIZON steps by repeating the last pose.
"""
import argparse
import asyncio
import functools
import http
import logging
import os
import sys
import time
import traceback

import msgpack
import numpy as np

logger = logging.getLogger("beingh_bridge")

# ----------------------------------------------------------------------------- msgpack-numpy (openpi wire format)
def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ----------------------------------------------------------------------------- rotation helpers (no scipy needed)
def quat_wxyz_normalize(q):
    q = np.asarray(q, dtype=np.float64).ravel()
    n = np.linalg.norm(q)
    q = q / n if n > 1e-12 else np.array([1.0, 0, 0, 0])
    return -q if q[0] < 0 else q          # canonical w>=0 (same as the converter)


def quat_wxyz_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_wxyz_to_rotvec(q):
    w = np.clip(q[0], -1.0, 1.0)
    v = q[1:]
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(s, w)
    return v / s * angle


def rotvec_to_matrix(v):
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def matrix_to_quat_wxyz(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S
    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0 else q


# ----------------------------------------------------------------------------- policy wrapper
class BeingHAmBenchPolicy:
    def __init__(self, args):
        from BeingH.inference.beingh_policy import BeingHPolicy
        from BeingH.utils.constants import INSTRUCTION_TEMPLATE

        template = "{task_description}" if args.prompt_template == "short" else INSTRUCTION_TEMPLATE
        self.policy = BeingHPolicy(
            model_path=args.ckpt,
            data_config_name=args.data_config_name,
            dataset_name=args.dataset_name,
            embodiment_tag=args.embodiment_tag,
            instruction_template=template,
            max_view_num=-1,
            use_fixed_view=False,
            device=args.device,
            num_inference_timesteps=args.num_inference_timesteps,
            enable_rtc=False,                       # we never feed prev_chunk; plain open-loop chunks
            metadata_variant=args.metadata_variant,
            stats_selection_mode=args.stats_selection_mode,
        )
        self.horizon = args.horizon
        self.gripper_mode = args.gripper_mode
        self.chunk = self.policy.action_chunk_length
        self.n = 0
        self.dt = []

    def metadata(self):
        return {"action_representation": "ee_absolute", "backend": "being-h05",
                "chunk_length": int(self.chunk), "horizon": int(self.horizon)}

    @staticmethod
    def _img(img):
        a = np.asarray(img)
        if a.dtype != np.uint8:
            a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        if a.ndim == 4:
            a = a[0]
        return np.ascontiguousarray(a)

    def infer(self, obs):
        t0 = time.perf_counter()
        prompt = obs.get("prompt", "") or ""
        if isinstance(prompt, bytes):
            prompt = prompt.decode()
        pos = np.asarray(obs["am_bench/ee_pos"], dtype=np.float64).ravel()[:3]
        quat = quat_wxyz_normalize(obs["am_bench/ee_quat"])
        grip = np.asarray(obs["am_bench/gripper_width"], dtype=np.float64).ravel()[:1]

        # same layout the official LIBERO client uses: batch dim of 1 + list instruction
        policy_obs = {
            "video.wrist_view": self._img(obs["am_bench/ee_image"])[None],           # (1,H,W,3) uint8
            "state.eef_position": pos.astype(np.float32).reshape(1, 3),
            "state.eef_rotation": quat_wxyz_to_rotvec(quat).astype(np.float32).reshape(1, 3),
            "state.gripper": grip.astype(np.float32).reshape(1, 1),
            "language.instruction": [prompt],
        }
        out = self.policy.get_action(policy_obs)
        d_pos = np.asarray(out["action.eef_position"], dtype=np.float64).reshape(-1, 3)
        d_rot = np.asarray(out["action.eef_rotation"], dtype=np.float64).reshape(-1, 3)
        g_cmd = np.asarray(out["action.gripper_position"], dtype=np.float64).reshape(-1)
        n = d_pos.shape[0]

        # integrate EE-local deltas -> absolute world poses (chained), same convention as
        # BeingH/benchmark/utils/policy.py "eef_delta": p' = p + R d_pos ; R' = R * R(d_rot)
        R = quat_wxyz_to_matrix(quat)
        p = pos.copy()
        acts = np.zeros((self.horizon, 8), dtype=np.float32)
        for i in range(self.horizon):
            j = min(i, n - 1)
            if i < n:
                p = p + R @ d_pos[j]
                R = R @ rotvec_to_matrix(d_rot[j])
            acts[i, :3] = p
            acts[i, 3:7] = matrix_to_quat_wxyz(R)
            g = g_cmd[j]
            acts[i, 7] = (1.0 if g > 0 else -1.0) if self.gripper_mode == "sign" else float(np.clip(g, -1, 1))

        if not np.all(np.isfinite(acts)):
            logger.warning("non-finite values in action chunk; replacing with current pose")
            acts = np.nan_to_num(acts, nan=0.0, posinf=0.0, neginf=0.0)
        dt = (time.perf_counter() - t0) * 1000
        self.n += 1
        self.dt.append(dt)
        if self.n <= 3 or self.n % 50 == 0:
            logger.info("infer #%d: %.0f ms, chunk %d -> %s, |d_pos| mean %.4f m, first %s",
                        self.n, dt, n, acts.shape, float(np.linalg.norm(d_pos, axis=1).mean()),
                        np.round(acts[0], 4).tolist())
        return {"actions": acts}


# ----------------------------------------------------------------------------- websocket server (openpi-compatible)
class WebsocketPolicyServer:
    def __init__(self, policy, host, port, metadata):
        self._policy, self._host, self._port, self._metadata = policy, host, port, metadata

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        import websockets.asyncio.server as _server
        async with _server.serve(self._handler, self._host, self._port, compression=None, max_size=None,
                                 process_request=_health_check, ping_interval=None, ping_timeout=None) as server:
            logger.info("SERVER_READY: listening on %s:%d", self._host, self._port)
            print(f"SERVER_READY {self._port}", flush=True)
            await server.serve_forever()

    async def _handler(self, websocket):
        import websockets
        import websockets.frames
        logger.info("connection from %s opened", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))
        prev_total = None
        while True:
            try:
                start = time.monotonic()
                obs = unpackb(await websocket.recv())
                t_inf = time.monotonic()
                action = self._policy.infer(obs)
                action["server_timing"] = {"infer_ms": (time.monotonic() - t_inf) * 1000}
                if prev_total is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total * 1000
                await websocket.send(packer.pack(action))
                prev_total = time.monotonic() - start
            except websockets.ConnectionClosed:
                logger.info("connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR,
                                      reason="Internal server error. Traceback included in previous frame.")
                raise


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Being-H checkpoint dir (contains model.safetensors, config.json, <dataset>_metadata.json)")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--data-config-name", default="ambench_fahexa")
    ap.add_argument("--dataset-name", default="ambench_posttrain", help="group name in the training yaml / DATASET_INFO")
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--metadata-variant", default=None, help="e.g. mt7 / smoke (task-level stats) or an embodiment name")
    ap.add_argument("--stats-selection-mode", default="auto", choices=["auto", "task", "embodiment"])
    ap.add_argument("--prompt-template", default="long", choices=["long", "short"], help="must match training (--prompt_template)")
    ap.add_argument("--num-inference-timesteps", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=50, help="AM-Bench expects (50, 8)")
    ap.add_argument("--gripper-mode", default="sign", choices=["sign", "raw"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [beingh] %(message)s")
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not os.path.isdir(args.ckpt):
        sys.exit(f"ckpt dir not found: {args.ckpt}")
    policy = BeingHAmBenchPolicy(args)
    WebsocketPolicyServer(policy, args.host, args.port, policy.metadata()).serve_forever()


if __name__ == "__main__":
    main()
