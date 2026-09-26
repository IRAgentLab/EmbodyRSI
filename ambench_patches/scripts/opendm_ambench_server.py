#!/usr/bin/env python3
"""OpenDM DM0.5 (AM-Bench LoRA checkpoint) <-> AM-Bench openpi websocket bridge.

Protocol (same as env/cosmos_ambench_server.py / env/lingbot_bridge.py): one
websocket port, msgpack-numpy frames, metadata first, then
``infer(observation) -> {"actions": (50, 8)}``.

Observation (from ambench_learn.policies.pi.eval):
    am_bench/ee_image      HxWx3 uint8   wrist camera  -> the single "Wrist" image
    am_bench/base_image    HxWx3 uint8   third person  -> ignored (not in training data)
    am_bench/ee_pos        (3,)          world xyz
    am_bench/ee_quat       (4,) wxyz
    am_bench/gripper_width (1,) metres
    prompt                 str
Output: absolute EE targets, 8 dims = [xyz, quat wxyz, gripper command in [-1, 1]]
(``action_representation = ee_absolute``).

How it maps onto OpenDM (read from opendm/exp/dm05_exp.py + opendm/data/transforms.py):
* We run the OpenDM inference runtime *in-process* (``DM05Exp._initialize_inference_runtime``)
  instead of proxying to its Flask ``/v1/infer``: no base64/JSON hop, one process, one GPU.
  ``DM05InferenceConfig._prepare_model_input`` + ``_predict`` are exactly what the
  Flask handler calls.
* state = [xyz, rotvec(quat wxyz), gripper_width] with the same quat->rotvec as the
  converter (env/convert_opendm.py::quat_wxyz_to_axis_angle == transforms._quat_to_rotvec).
* robot_type "UR5": the registration borrows RobotType.UR5 ([EEF]*6+[GRIPPER]); the
  request must carry it so ``_resolve_state_desc`` and the per-robot norm-stats profile
  ("UR5" in norm_stats.json) match training.
* Training was RELATIVE: action - state element-wise on xyz+axis-angle, gripper absolute.
  The runtime inverts it with ``ActionAbsolute(compose_eef_rot=False)`` (state + delta),
  and ``ArrangeState`` (rpy -> axis-angle rewrite) stays off because compose_eef_rot is
  False and we already send axis-angle. Do NOT switch compose_eef_rot on: that would
  quaternion-compose deltas that were trained as plain differences.
* Output gripper = model action[6] (trained on the raw +-1 AM-Bench command), clipped.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import http
import importlib.util
import logging
import math
import os
import sys
import time
import traceback

import msgpack
import numpy as np
from PIL import Image

logger = logging.getLogger("opendm_bridge")

A = os.environ.get("AMBENCH_ROOT", "/public/home/liaodl/ambench")
OPENDM_ROOT = os.environ.get("OPENDM_ROOT", f"{A}/env/opendm/opendm")
if OPENDM_ROOT not in sys.path:
    sys.path.insert(0, OPENDM_ROOT)
os.environ.setdefault("OPENDM_DATA_PATH", f"{A}/env/opendm/registry")


# ----------------------------------------------------------------------------- msgpack-numpy
# Copied from openpi (lingbot-va vendored copy: wan_va/utils/Simple_Remote_Infer/deploy/
# msgpack_numpy.py) so this venv does not depend on openpi / lingbot-va.
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


# ----------------------------------------------------------------------------- websocket server
class WebsocketPolicyServer:
    """openpi websocket_policy_server wire protocol; ping disabled so a slow first
    (CUDA-graph capturing) inference cannot trip the keepalive."""

    def __init__(self, policy, host="0.0.0.0", port=None, metadata=None):
        self._policy, self._host, self._port = policy, host, port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        import websockets.asyncio.server as _server

        async with _server.serve(
            self._handler, self._host, self._port, compression=None, max_size=None,
            process_request=_health_check, ping_interval=None, ping_timeout=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        import websockets
        import websockets.frames

        logger.info("Connection from %s opened", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))
        prev_total = None
        while True:
            try:
                t_start = time.monotonic()
                obs = unpackb(await websocket.recv())
                t_inf = time.monotonic()
                action = self._policy.infer(obs)
                action["server_timing"] = {"infer_ms": (time.monotonic() - t_inf) * 1000}
                if prev_total is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total * 1000
                await websocket.send(packer.pack(action))
                prev_total = time.monotonic() - t_start
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


# ----------------------------------------------------------------------------- rotations
def quat_wxyz_to_rotvec(q) -> np.ndarray:
    """Same algorithm as env/convert_opendm.py / opendm transforms._quat_to_rotvec (w>=0, |v|<=pi)."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-8:
        raise ValueError(f"degenerate quaternion {q}")
    q = q / n
    if q[0] < 0:
        q = -q
    vec = q[1:4]
    vn = np.linalg.norm(vec)
    if vn < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return vec * (2.0 * math.atan2(vn, q[0]) / vn)


def rotvec_to_quat_wxyz(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(v)
    if theta < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = v / theta
    return np.concatenate([[math.cos(theta / 2)], axis * math.sin(theta / 2)])


# ----------------------------------------------------------------------------- policy
def _load_playground(path: str):
    spec = importlib.util.spec_from_file_location("dm05_ambench_lora", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class OpenDMAmBenchPolicy:
    def __init__(self, ckpt: str, dataset: str, chunk: int, diffusion_steps: int,
                 playground: str, norm_stats_root: str | None, robot_type: str):
        mod = _load_playground(playground)
        exp = mod.DM05Exp(task="inference", use_lora=True)
        exp.model_config.model_name_or_path = ckpt
        exp.model_config.chunk_size = chunk
        exp.data_config.dataset_name = dataset
        if norm_stats_root:
            exp.data_config.norm_stats_root = norm_stats_root
        exp.inference_config.output_action_dim = 7
        exp.inference_config.image_prompts = ["Wrist"]
        exp.inference_config.compose_eef_rot = False
        exp.inference_config.diffusion_steps = diffusion_steps
        assert exp.data_config.action_mode.value == "relative", exp.data_config.action_mode
        logger.info("loading OpenDM runtime: ckpt=%s dataset=%s chunk=%d steps=%d",
                    ckpt, dataset, chunk, diffusion_steps)
        t0 = time.time()
        exp._initialize_inference_runtime()
        self.exp = exp
        self.ic = exp.inference_config
        self.robot_type = robot_type
        self.chunk = chunk
        self._n = 0
        logger.info("OpenDM runtime ready in %.0fs (default_robot_type=%s, state_desc=%s)",
                    time.time() - t0, self.ic.default_robot_type, self.ic.default_state_desc)

    def metadata(self) -> dict:
        return {"action_representation": "ee_absolute", "backend": "opendm-dm05-lora",
                "chunk": self.chunk, "action_dim": 8}

    @staticmethod
    def _img(x) -> Image.Image:
        img = np.asarray(x)
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        return Image.fromarray(np.ascontiguousarray(img[..., :3]), "RGB")

    def infer(self, obs: dict) -> dict:
        t0 = time.perf_counter()
        pos = np.asarray(obs["am_bench/ee_pos"], dtype=np.float64).ravel()[:3]
        quat = np.asarray(obs["am_bench/ee_quat"], dtype=np.float64).ravel()[:4]
        grip = float(np.asarray(obs["am_bench/gripper_width"], dtype=np.float64).ravel()[0])
        state = np.concatenate([pos, quat_wxyz_to_rotvec(quat), [grip]]).astype(np.float32)
        prompt = obs.get("prompt", "") or ""
        if isinstance(prompt, bytes):
            prompt = prompt.decode()

        data = self.ic._prepare_model_input(
            text=prompt, images=[self._img(obs["am_bench/ee_image"])],
            states=[float(v) for v in state], robot_type=self.robot_type,
        )
        act = np.asarray(self.ic._predict(data), dtype=np.float64)  # (chunk, 7) absolute
        if act.ndim != 2 or act.shape[1] != 7:
            raise RuntimeError(f"unexpected OpenDM action shape {act.shape}")

        out = np.zeros((act.shape[0], 8), dtype=np.float32)
        out[:, :3] = act[:, :3]
        for i in range(act.shape[0]):
            out[i, 3:7] = rotvec_to_quat_wxyz(act[i, 3:6])
        out[:, 7] = np.clip(act[:, 6], -1.0, 1.0)
        if not np.all(np.isfinite(out)):
            raise RuntimeError("non-finite action from OpenDM")

        self._n += 1
        dt = (time.perf_counter() - t0) * 1000
        if self._n <= 3 or self._n % 20 == 0:
            logger.info("infer #%d %.0f ms (model %.0f ms) prompt=%r state=%s -> actions %s first=%s",
                        self._n, dt, (self.ic.last_model_latency_sec or 0) * 1000, prompt,
                        np.round(state, 4).tolist(), out.shape, np.round(out[0], 4).tolist())
        return {"actions": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-port", type=int, required=True)
    ap.add_argument("--ckpt", required=True, help="LoRA step checkpoint dir (has adapter_config.json + norm_stats.json)")
    ap.add_argument("--dataset", default="ambench_mt7", help="registered dataset name used in training")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--diffusion-steps", type=int, default=10)
    ap.add_argument("--robot-type", default="UR5")
    ap.add_argument("--norm-stats-root", default=os.environ.get("OPENDM_NORM_STATS"))
    ap.add_argument("--playground", default=f"{OPENDM_ROOT}/playground/dm05_ambench_lora.py")
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [opendm] %(message)s")

    policy = OpenDMAmBenchPolicy(args.ckpt, args.dataset, args.chunk, args.diffusion_steps,
                                 args.playground, args.norm_stats_root, args.robot_type)
    logger.info("服务就绪：监听 %s:%d", args.host, args.listen_port)
    WebsocketPolicyServer(policy=policy, host=args.host, port=args.listen_port,
                          metadata=policy.metadata()).serve_forever()


if __name__ == "__main__":
    main()
