"""WipeWindow expert: force-servo pressing. Position-only pressing cannot regulate the contact: 0.215 m depth gave 17-35 N
(sponge pivots out of the pinch), 0.225 m gave 0.1 N (no contact). During each press/sweep window the x target is now
adjusted per step from the measured sponge-window force toward AMBENCH_WIPE_FDES (default 4 N)."""
import pathlib, shutil
p = pathlib.Path("/public/home/liaodl/ambench/src/ambench/source/ambench/ambench/policies/scripted/wipe_window.py")
s = p.read_text()
bak = p.with_suffix(".py.bak_servo"); bak.exists() or shutil.copy(p, bak)
# 1. default contact depth: start just short of contact, the servo adds depth as needed
old = 'contact[0] = dot_pos[0] - float(os.environ.get("AMBENCH_WIPE_CONTACT", "0.15"))'
assert s.count(old) == 1
s = s.replace(old, 'contact[0] = dot_pos[0] - float(os.environ.get("AMBENCH_WIPE_CONTACT", "0.24"))  # just short of contact; force servo adds depth')
# 2. record press windows [t(close waypoint reached) .. t(end of sweep pause)]
old = "            current_time += making_contact_time\n"
assert s.count(old) == 1
s = s.replace(old, "            press_t0 = current_time  # from the 'close' pose onward the force servo is active\n" + old)
old = ("            current_time += pause_time\n"
       "            waypoints.append(Waypoint(t=current_time, xyz=sweep, quat=identity_quat, gripper=-1.0))\n")
assert s.count(old) == 1
s = s.replace(old, old + "            self._press_windows.append((press_t0, current_time))\n")
# ensure _press_windows exists before the loop
old2 = '        zoff = float(os.environ.get("AMBENCH_WIPE_ZOFF", "0.088"))'
assert s.count(old2) == 1
s = s.replace(old2, '        self._press_windows = []\n        self._press_dx = 0.0\n        zoff = float(os.environ.get("AMBENCH_WIPE_ZOFF", "0.0"))')
# 3. advance override with the force servo
old = "    def generate_trajectory(self, obs, env_id: int = 0):"
assert s.count(old) == 1
s = s.replace(old, '''    def _sponge_force(self, env_id: int) -> float:
        env = cast("WipeWindow", self.env)
        try:
            return float(torch.norm(env.contact_sensor.data.net_forces_w[env_id, 0, :]).item())
        except Exception:
            return 0.0

    def advance(self, obs, env_id) -> torch.Tensor:
        """Base waypoint tracking plus a force servo on the window-normal (x) axis while pressing."""
        action = super().advance(obs, env_id)
        t = self.step_count - 1
        windows = getattr(self, "_press_windows", [])
        if os.environ.get("AMBENCH_WIPE_FSERVO", "1") == "1" and any(t0 <= t <= t1 for t0, t1 in windows):
            f = self._sponge_force(env_id)
            f_des = float(os.environ.get("AMBENCH_WIPE_FDES", "4.0"))
            step = float(os.environ.get("AMBENCH_WIPE_FSTEP", "0.0005"))  # m per sim step
            dx = getattr(self, "_press_dx", 0.0)
            if f < f_des - 1.0:
                dx += step
            elif f > f_des + 1.0:
                dx -= step
            self._press_dx = max(-0.03, min(0.06, dx))
            action[0] = action[0] + self._press_dx
        else:
            self._press_dx = 0.0
        return action

    def generate_trajectory(self, obs, env_id: int = 0):''')
if "\nimport torch\n" not in s:
    s = s.replace("\nimport os\n", "\nimport os\nimport torch\n", 1)
    assert "\nimport torch\n" in s
p.write_text(s); print("patched wipe_window.py: force servo, CONTACT default 0.24, ZOFF default 0")
