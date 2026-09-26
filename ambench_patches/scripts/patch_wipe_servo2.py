"""WipeWindow force servo: use the wrist wrench x (pressing axis, ee frame) as the default force signal. Verified against the
sponge contact sensor (18.7 N vs 18.8 N while only the sponge touches); unlike the sponge sensor it also sees the gripper
itself hitting the window (sponge sensor 0 N while wrist reads 40 N), which would otherwise drive the servo deeper."""
import pathlib
p = pathlib.Path("/public/home/liaodl/ambench/src/ambench/source/ambench/ambench/policies/scripted/wipe_window.py")
s = p.read_text()
old = '''        if os.environ.get("AMBENCH_WIPE_FSERVO", "1") == "1" and any(t0 <= t <= t1 for t0, t1 in windows):
            f = self._sponge_force(env_id)'''
new = '''        if os.environ.get("AMBENCH_WIPE_FSERVO", "1") == "1" and any(t0 <= t <= t1 for t0, t1 in windows):
            f = self._press_force(obs, env_id)'''
assert s.count(old) == 1; s = s.replace(old, new)
old = '''    def advance(self, obs, env_id) -> torch.Tensor:
        """Base waypoint tracking plus a force servo'''
new = '''    def _press_force(self, obs, env_id: int) -> float:
        """Pressing force along the EE x axis. Default: wrist wrench (ee_wrench[0]); AMBENCH_WIPE_FSRC=sponge uses the
        sponge contact sensor instead (blind to the gripper itself touching the window)."""
        if os.environ.get("AMBENCH_WIPE_FSRC", "wrist") == "sponge":
            return self._sponge_force(env_id)
        try:
            return float(obs["policy"][env_id]["ee_wrench"][0].item())
        except Exception:
            return self._sponge_force(env_id)

    def advance(self, obs, env_id) -> torch.Tensor:
        """Base waypoint tracking plus a force servo'''
assert s.count(old) == 1; s = s.replace(old, new)
p.write_text(s); print("patched: servo force source = wrist wrench x (AMBENCH_WIPE_FSRC=sponge to switch)")
