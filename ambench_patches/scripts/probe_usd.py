"""Print collision approximation / physics material / bbox for the task assets (run inside the Isaac container)."""
import sys
try:
    from pxr import Usd, UsdGeom, UsdShade
except ModuleNotFoundError:
    import isaacsim  # sets up Kit's python libs
    from pxr import Usd, UsdGeom, UsdShade
root = "/public/home/liaodl/ambench/src/ambench/source/ambench/ambench/assets/"
for f in ["robots/hexa_scorpion.usd"]:
    print("\n---", f)
    st = Usd.Stage.Open(root + f)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    for p in Usd.PrimRange(st.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        name = p.GetName()
        if "finger" not in str(p.GetPath()).lower() and "hexa_scorpion" in f and p.GetTypeName() not in ("PhysicsPrismaticJoint",):
            continue
        out = []
        for a in p.GetAttributes():
            n = a.GetName()
            if n.startswith(("physics:", "physxCollision:", "physxRigidBody:", "drive:", "physxJoint:")) and a.HasAuthoredValue():
                v = a.Get()
                if n in ("physics:body0", "physics:body1"):
                    continue
                out.append(f"{n}={v}")
        api = p.GetAppliedSchemas()
        api = [s for s in api if "Physics" in s or "Collision" in s or "Material" in s]
        if p.HasAPI(UsdShade.MaterialBindingAPI):
            for purpose in ("physics", ""):
                b = UsdShade.MaterialBindingAPI(p).GetDirectBinding(materialPurpose=purpose)
                if b.GetMaterial():
                    out.append(f"mat[{purpose or 'all'}]={b.GetMaterialPath()}")
        ext = ""
        if p.IsA(UsdGeom.Boundable):
            bb = cache.ComputeLocalBound(p).ComputeAlignedRange()
            mn, mx = bb.GetMin(), bb.GetMax()
            ext = f" bbox=[{mn[0]:.3f},{mx[0]:.3f}]x[{mn[1]:.3f},{mx[1]:.3f}]x[{mn[2]:.3f},{mx[2]:.3f}]"
            if p.IsA(UsdGeom.Mesh):
                pts = p.GetAttribute("points").Get()
                ext += f" npts={len(pts) if pts else 0}"
        if out or ext or api:
            print(p.GetPath(), p.GetTypeName(), ",".join(api), " ".join(out) + ext)
