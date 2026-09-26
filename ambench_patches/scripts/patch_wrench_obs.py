"""Add an opt-in wrist F/T observation ('ee_wrench', 6-dim, ee_link body frame) to every task env, and a WRENCH=1 knob
to the recording job that appends it to the recorded state vector."""
import pathlib, shutil
R = pathlib.Path("/public/home/liaodl/ambench/src/ambench/source/ambench/ambench")
p = R / "tasks/base/base_env.py"; s = p.read_text()
bak = p.with_suffix(".py.bak_wrench"); bak.exists() or shutil.copy(p, bak)
old_step = '''    def step(self, action: torch.Tensor):
        """Apply observation noise once to list or tensor policy observations."""'''
new_step = '''    def _wrist_wrench_b(self) -> torch.Tensor | None:
        """Wrist F/T sensor equivalent: reaction wrench of the fixed joint that carries ee_link, in the ee_link frame.

        Includes everything downstream of the wrist: contact forces transmitted through a held tool (e.g. the sponge),
        the held object's weight and inertial load. Shape (num_envs, 6) = [force_xyz, torque_xyz].
        """
        if getattr(self, "ee_link_idx", None) is None:
            return None
        try:
            return self.robot.data.body_incoming_joint_wrench_b[:, self.ee_link_idx]
        except Exception:
            return None

    def _inject_extra_obs(self, obs_buf):
        """Append 'ee_wrench' to each per-env policy observation dict (opt-in for recording via state_keys)."""
        policy_obs = obs_buf.get("policy") if isinstance(obs_buf, dict) else None
        if not isinstance(policy_obs, list) or not policy_obs:
            return obs_buf
        wrench = self._wrist_wrench_b()
        if wrench is None:
            return obs_buf
        for env_idx, env_obs in enumerate(policy_obs):
            if isinstance(env_obs, dict) and "ee_wrench" not in env_obs:
                env_obs["ee_wrench"] = wrench[env_idx].clone()
        return obs_buf

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs_buf, extras = super().reset(seed=seed, options=options)
        return self._inject_extra_obs(obs_buf), extras

    def step(self, action: torch.Tensor):
        """Apply observation noise once to list or tensor policy observations."""'''
assert s.count(old_step) == 1
s = s.replace(old_step, new_step)
old_ret = '''            elif isinstance(policy_obs, torch.Tensor):
                obs_buf["policy"] = self._observation_noise_model(policy_obs)

        return obs_buf, reward_buf, terminated, truncated, extras'''
new_ret = '''            elif isinstance(policy_obs, torch.Tensor):
                obs_buf["policy"] = self._observation_noise_model(policy_obs)

        self._inject_extra_obs(obs_buf)  # after noise: the wrist wrench is not part of the noise model
        return obs_buf, reward_buf, terminated, truncated, extras'''
assert s.count(old_ret) == 1
s = s.replace(old_ret, new_ret); p.write_text(s); print("patched base_env.py")

# recording job: WRENCH=1 appends ee_wrench to the state keys
p = pathlib.Path("/public/home/liaodl/ambench/env/record_hpc.sbatch"); s = p.read_text()
old = '  SKEYS="${STATEKEYS:-ee_pos ee_quat gripper_width}"\n'
assert s.count(old) == 1
s = s.replace(old, old + '[ "${WRENCH:-0}" = "1" ] && SKEYS="$SKEYS ee_wrench"   # 腕部六维力（ee_link 固定关节反作用力，机体系）追加进 state\n')
p.write_text(s); print("patched record_hpc.sbatch (WRENCH=1)")

p = pathlib.Path("/public/home/liaodl/ambench/src/ambench/scripts/data/export_lerobot_to_openpi.py"); s = p.read_text()
bak = p.with_suffix(".py.bak_wrench"); bak.exists() or shutil.copy(p, bak)
old = '    "base_quat": 4,\n}\n'
assert s.count(old) == 1
s = s.replace(old, '    "base_quat": 4,\n    "ee_wrench": 6,  # wrist F/T (ee_link joint reaction wrench, body frame); exported only with --include_wrench\n}\n')
old = '        frame.update({key: state_parts[key].astype(np.float32, copy=False) for key in required})\n    elif action_semantics == BASE_JOINT_ABSOLUTE:'
assert s.count(old) == 1
s = s.replace(old, '        frame.update({key: state_parts[key].astype(np.float32, copy=False) for key in required})\n'
                   '        if include_wrench and "ee_wrench" in state_parts:\n'
                   '            frame["ee_wrench"] = state_parts["ee_wrench"].astype(np.float32, copy=False)\n'
                   '    elif action_semantics == BASE_JOINT_ABSOLUTE:')
old = '    include_base_image: bool,\n'
assert s.count(old) == 1
s = s.replace(old, '    include_base_image: bool,\n    include_wrench: bool = False,\n')
old = '                    include_base_image=not args.omit_base_image,\n'
assert s.count(old) == 1
s = s.replace(old, old + '                    include_wrench=args.include_wrench,\n')
old = '    parser.add_argument("--target_hz", type=int, default=20, help="Logical policy FPS for the OpenPI export.")\n'
assert s.count(old) == 1
s = s.replace(old, old + '    parser.add_argument("--include_wrench", action="store_true", help="Also export the recorded wrist wrench (ee_wrench, 6D) as a frame feature.")\n')
p.write_text(s); print("patched export_lerobot_to_openpi.py (ee_wrench, --include_wrench)")
