"""Give the frame's grab handle (frame_body/Cube_04) a box collider; keep a backup of the original asset."""
import shutil, pathlib
try:
    from pxr import Usd, UsdPhysics
except ModuleNotFoundError:
    import isaacsim
    from pxr import Usd, UsdPhysics
f = pathlib.Path("/public/home/liaodl/ambench/src/ambench/source/ambench/ambench/assets/objects/frame.usd")
bak = f.with_suffix(".usd.orig")
if not bak.exists():
    shutil.copy(f, bak)
st = Usd.Stage.Open(str(f))
prim = st.GetPrimAtPath("/World/frame/frame_body/Cube_04")
assert prim.IsValid(), "Cube_04 not found"
print("before:", prim.HasAPI(UsdPhysics.CollisionAPI))
UsdPhysics.CollisionAPI.Apply(prim)
prim.GetAttribute("physics:collisionEnabled").Set(True)
st.GetRootLayer().Save()
st2 = Usd.Stage.Open(str(f))
p2 = st2.GetPrimAtPath("/World/frame/frame_body/Cube_04")
print("after:", p2.HasAPI(UsdPhysics.CollisionAPI), p2.GetAttribute("physics:collisionEnabled").Get())
