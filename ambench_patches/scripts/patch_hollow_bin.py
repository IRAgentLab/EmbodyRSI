"""Hollow bin at runtime: the payload bin mesh is a kinematic body whose mesh collider PhysX treats as a convex hull
(solid block), and file-level overrides did not compose reliably. In _setup_scene, disable the mesh collider and add
five exact box colliders (bottom + 4 walls) under each env's container prim."""
import pathlib, shutil
R = pathlib.Path("/public/home/liaodl/ambench/src/ambench/source/ambench/ambench")
# 1. helper in base_env
p = R / "tasks/base/base_env.py"; s = p.read_text()
old = "    def _wrist_wrench_b(self) -> torch.Tensor | None:"
new = '''    def _make_hollow_bin(self, prim_path_expr: str, size: tuple[float, float, float], wall: float = 0.01) -> int:
        """Replace a bin's mesh collider by exact box colliders (bottom + 4 walls) so objects can enter it.

        The NVIDIA container asset is a kinematic rigid body with a plain mesh collider; PhysX cooks that as a convex
        hull, i.e. a solid block. Must run before the simulation starts (called from _setup_scene).
        """
        import isaaclab.sim as sim_utils
        from omni.usd import get_context
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = get_context().get_stage()
        sx, sy, sz = size
        boxes = {  # name: (center xyz in the container frame, full extents)
            "col_bottom": ((0.0, 0.0, wall / 2), (sx, sy, wall)),
            "col_xneg": ((-sx / 2 + wall / 2, 0.0, sz / 2), (wall, sy, sz)),
            "col_xpos": ((sx / 2 - wall / 2, 0.0, sz / 2), (wall, sy, sz)),
            "col_yneg": ((0.0, -sy / 2 + wall / 2, sz / 2), (sx, wall, sz)),
            "col_ypos": ((0.0, sy / 2 - wall / 2, sz / 2), (sx, wall, sz)),
        }
        n = 0
        for root_path in sim_utils.find_matching_prim_paths(prim_path_expr):
            root = stage.GetPrimAtPath(root_path)
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsInstanceProxy():
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            for name, (center, ext) in boxes.items():
                cube = UsdGeom.Cube.Define(stage, f"{root_path}/{name}")
                cube.GetSizeAttr().Set(1.0)
                cube.GetPurposeAttr().Set(UsdGeom.Tokens.guide)  # collider only, not rendered
                xf = UsdGeom.Xformable(cube.GetPrim())
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(*center))
                xf.AddScaleOp().Set(Gf.Vec3f(*ext))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            n += 1
        return n

    def _wrist_wrench_b(self) -> torch.Tensor | None:'''
assert s.count(old) == 1; s = s.replace(old, new); p.write_text(s); print("patched base_env: _make_hollow_bin")
# 2. use it in TossBall and LemonHarvesting
for rel, anchor in [("tasks/toss_ball/toss_ball_env.py", '        self.scene.rigid_objects["container"] = self.container\n'),
                    ("tasks/lemon_harvesting/lemon_harvesting_env.py", None)]:
    p = R / rel; s = p.read_text()
    if anchor is None:
        import re
        m = re.search(r'        self\.scene\.rigid_objects\["container"\] = self\.container\n', s)
        if not m:
            print("lemon: container registration line not found, skipped"); continue
        anchor = m.group(0)
    assert s.count(anchor) == 1, rel
    s = s.replace(anchor, anchor + '        n_bins = self._make_hollow_bin(self.cfg.container_object_cfg.prim_path, tuple(self.cfg.container_size))\n        print(f"[hollow-bin] replaced the bin mesh collider by box colliders in {n_bins} envs", flush=True)\n')
    p.write_text(s); print("patched", rel)
