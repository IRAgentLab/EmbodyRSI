"""Smoke test for the opt-in 'ee_wrench' observation: run the WipeWindow expert and print the wrist wrench next to the
sponge contact-sensor force so the two can be compared, plus the obs keys the recorder would see."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="WipeWindow-Am-FAHexa-Abs-PID-Direct-v0")
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--every", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.headless = True; args.enable_cameras = True
app = AppLauncher(args).app
import importlib, gymnasium as gym, torch, isaaclab_tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import ambench.tasks  # noqa
env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
env = gym.make(args.task, cfg=env_cfg).unwrapped
obs, _ = env.reset()
print("[obs keys after reset]", sorted(obs["policy"][0].keys()), flush=True)
print("[ee_wrench@reset]", obs["policy"][0].get("ee_wrench"), flush=True)
ep = gym.spec(args.task).kwargs["scripted_policy_entry_point"]; mod, cls = ep.split(":")
policy = getattr(importlib.import_module(mod), cls)(env=env, inject_noise=False); policy.reset()
with torch.inference_mode():
    for step in range(1, args.steps + 1):
        obs, *_ = env.step(policy.advance(obs, 0))
        if step % args.every == 0:
            w = obs["policy"][0]["ee_wrench"]
            fs = torch.norm(env.contact_sensor.data.net_forces_w[0, 0, :]).item() if hasattr(env, "contact_sensor") else float("nan")
            print(f"[cmp] t={step:4d} sponge_sensor|F|={fs:6.2f}N  wrist_F=({w[0]:7.2f},{w[1]:7.2f},{w[2]:7.2f}) |F|={torch.norm(w[:3]).item():6.2f}N  wrist_M=({w[3]:6.3f},{w[4]:6.3f},{w[5]:6.3f})", flush=True)
env.close(); app.close()
