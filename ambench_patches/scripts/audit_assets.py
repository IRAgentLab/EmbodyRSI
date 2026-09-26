"""Audit every task asset: geometry prims without colliders, rigid-body mass, physics material, collision approximation."""
import glob, os
try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
except ModuleNotFoundError:
    import isaacsim
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
root = "/public/home/liaodl/ambench/src/ambench/source/ambench/ambench/assets/"
files = sorted(glob.glob(root + "objects/*.usd")) + sorted(glob.glob(root + "objects/**/*.usd", recursive=True))
seen = set()
for f in files:
    if f in seen: continue
    seen.add(f)
    st = Usd.Stage.Open(f)
    print(f"\n### {os.path.relpath(f, root)}")
    for p in Usd.PrimRange(st.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        tn = p.GetTypeName()
        geom = tn in ("Mesh", "Cube", "Cylinder", "Sphere", "Capsule", "Cone")
        rb = p.HasAPI(UsdPhysics.RigidBodyAPI)
        if not geom and not rb: continue
        flags = []
        if rb:
            m = p.GetAttribute("physics:mass").Get() if p.HasAPI(UsdPhysics.MassAPI) else None
            flags.append(f"RIGIDBODY mass={m}")
        if geom:
            has_col = p.HasAPI(UsdPhysics.CollisionAPI)
            # collider may be on an ancestor
            anc = p.GetParent(); anc_col = False
            while anc and anc.GetPath() != "/":
                if anc.HasAPI(UsdPhysics.CollisionAPI): anc_col = True; break
                anc = anc.GetParent()
            flags.append("collider" if has_col else ("collider(ancestor)" if anc_col else "**NO COLLIDER**"))
            ap = p.GetAttribute("physics:approximation").Get() if p.HasAPI(UsdPhysics.MeshCollisionAPI) else None
            if ap: flags.append(f"approx={ap}")
            mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial(materialPurpose="physics")
            if mat:
                mp = mat.GetPrim()
                flags.append(f"physmat(s={mp.GetAttribute('physics:staticFriction').Get()},d={mp.GetAttribute('physics:dynamicFriction').Get()})")
            bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeLocalBound(p).ComputeAlignedRange()
            mn, mx = bb.GetMin(), bb.GetMax()
            flags.append(f"bbox=[{mn[0]:.3f},{mx[0]:.3f}]x[{mn[1]:.3f},{mx[1]:.3f}]x[{mn[2]:.3f},{mx[2]:.3f}]")
        print(f"  {p.GetPath()} {tn}: " + " ".join(flags))
