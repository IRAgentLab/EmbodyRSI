"""container.usd: the bin mesh comes from a Nucleus payload and is a kinematic rigid body with a plain mesh collider,
which PhysX turns into a convex hull (a solid block): the ball bounces off the rim plane and can never enter.
Author an override on the payload's mesh prim that requests convex decomposition (hollow bin)."""
import shutil, pathlib
try:
    from pxr import Usd, UsdPhysics, Sdf
except ModuleNotFoundError:
    import isaacsim
    from pxr import Usd, UsdPhysics, Sdf
f = pathlib.Path("/public/home/liaodl/ambench/src/ambench/source/ambench/ambench/assets/objects/container.usd")
bak = f.with_suffix(".usd.orig")
if not bak.exists():
    shutil.copy(f, bak)
st = Usd.Stage.Open(str(f), Usd.Stage.LoadNone)
mesh_path = "/Root/Container/Container/Container_B06_01"   # payload default prim 'Container' -> mesh 'Container_B06_01'
prim = st.OverridePrim(mesh_path)
UsdPhysics.CollisionAPI.Apply(prim)
api = UsdPhysics.MeshCollisionAPI.Apply(prim)
api.CreateApproximationAttr().Set(__import__("os").environ.get("BIN_APPROX", "none"))  # kinematic actors accept triangle meshes: exact hollow bin
st.GetRootLayer().Save()
st2 = Usd.Stage.Open(str(f), Usd.Stage.LoadNone)
p2 = st2.GetPrimAtPath(mesh_path)
print("override authored:", p2.IsValid(), p2.GetAppliedSchemas(), p2.GetAttribute("physics:approximation").Get())
