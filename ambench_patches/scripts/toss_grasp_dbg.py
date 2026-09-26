"""TossBall grasp debug: reset, drive with the scripted expert, print ball / finger geometry in the EE frame each 5 steps."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TossBall-Am-FAHexa-Abs-PID-Direct-v0")
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--every", type=int, default=5)
parser.add_argument("--finger-k", type=float, default=None, help="override finger drive stiffness")
parser.add_argument("--finger-d", type=float, default=None, help="override finger drive damping")
parser.add_argument("--solver-pos", type=int, default=None, help="robot articulation solver position iterations")
parser.add_argument("--solver-vel", type=int, default=None, help="robot articulation solver velocity iterations")
parser.add_argument("--ball-solver-pos", type=int, default=None)
parser.add_argument("--ball-mu", type=str, default=None, help="ball physics material static:dynamic friction")
parser.add_argument("--finger-mu", type=str, default=None, help="bind a physics material static:dynamic to the finger collision prims")
parser.add_argument("--grasp-x", type=float, default=None, help="override ball_grasp_offset_local x")
parser.add_argument("--motion", type=str, default=None, help="synthetic motion from the hold pose: transx|transy|transz|pitch|roll|yaw (amplitude via --amp, ramp over --ramp steps)")
parser.add_argument("--amp", type=float, default=0.3)
parser.add_argument("--ramp", type=int, default=120)
parser.add_argument("--hold", action="store_true", help="repeat the initial expert action (no motion)")
parser.add_argument("--finger-mass", type=float, default=None, help="override finger link mass (kg), inertia scaled accordingly")
parser.add_argument("--armature", type=float, default=None, help="finger joint armature")
parser.add_argument("--pgs", action="store_true", help="use PGS solver instead of TGS")
parser.add_argument("--ball-damping", type=float, default=None, help="ball linear+angular damping")
parser.add_argument("--ball-depen", type=float, default=None, help="ball max_depenetration_velocity")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import importlib
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
import isaaclab.utils.math as math_utils
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import ambench.tasks  # noqa: F401  (registers envs)

env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
asset = env_cfg.robot_profile.robot.asset
for jn in ("arm_lfinger_joint", "arm_rfinger_joint"):
    act = asset.actuators[jn]
    if args.finger_k is not None: act.stiffness = args.finger_k
    if args.finger_d is not None: act.damping = args.finger_d
    if args.armature is not None: act.armature = args.armature
    print(f"[cfg] {jn}: stiffness={act.stiffness} damping={act.damping} effort={act.effort_limit_sim}")
ap = asset.spawn.articulation_props
if args.solver_pos is not None: ap.solver_position_iteration_count = args.solver_pos
if args.solver_vel is not None: ap.solver_velocity_iteration_count = args.solver_vel
print(f"[cfg] robot solver pos/vel = {ap.solver_position_iteration_count}/{ap.solver_velocity_iteration_count}")
if args.grasp_x is not None:
    env_cfg.ball_grasp_offset_local = (args.grasp_x, 0.0, 0.0)
if args.pgs:
    env_cfg.sim.physx.solver_type = 0
    print("[cfg] solver_type =", env_cfg.sim.physx.solver_type)
rp = env_cfg.ball_cfg.spawn.rigid_props
if args.ball_damping is not None:
    rp.linear_damping = args.ball_damping; rp.angular_damping = args.ball_damping
if args.ball_solver_pos is not None: rp.solver_position_iteration_count = args.ball_solver_pos
if args.ball_depen is not None: rp.max_depenetration_velocity = args.ball_depen
if args.ball_mu:
    import isaaclab.sim as sim_utils
    sf, df = (float(x) for x in args.ball_mu.split(":"))
    env_cfg.ball_cfg.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=sf, dynamic_friction=df, restitution=0.0)
print(f"[cfg] ball physics_material: {env_cfg.ball_cfg.spawn.physics_material}")
env = gym.make(args.task, cfg=env_cfg).unwrapped
import isaaclab.sim as sim_utils
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics
import omni.usd
stage = omni.usd.get_context().get_stage()

def describe_physics_material(path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return f"{path}: <invalid prim>"
    out = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial(materialPurpose="physics")
        info = "<no physics material bound>"
        if p.HasAPI(UsdPhysics.MeshCollisionAPI):
            info = f"approx={UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get()} " + info
        else:
            info = "approx=<default: convexHull for kinematic/dynamic> " + info
        if mat:
            mp = mat.GetPrim()
            sf = mp.GetAttribute("physics:staticFriction").Get(); df = mp.GetAttribute("physics:dynamicFriction").Get()
            info = f"{mp.GetPath()} static={sf} dynamic={df}"
        ext = ""
        for m in Usd.PrimRange(p, Usd.TraverseInstanceProxies()):
            if m.IsA(UsdGeom.Mesh):
                pts = m.GetAttribute("points").Get()
                xf = UsdGeom.Xformable(m).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                link_xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                rel_xf = xf * link_xf.GetInverse()
                q = [rel_xf.Transform(pt) for pt in pts]
                xs=[v[0] for v in q]; ys=[v[1] for v in q]; zs=[v[2] for v in q]
                ext += f" mesh_bbox_link=[{min(xs):.3f},{max(xs):.3f}]x[{min(ys):.3f},{max(ys):.3f}]x[{min(zs):.3f},{max(zs):.3f}]"
        out.append(f"    {p.GetPath()} -> {info}{ext} schemas={p.GetAppliedSchemas()} approx_attr={p.GetAttribute('physics:approximation').Get() if p.GetAttribute('physics:approximation') else None} proxy={p.IsInstanceProxy()}")
    return "\n".join(out) if out else f"{path}: <no collision prims>"

_root = stage.GetPrimAtPath("/World/envs/env_0/Container")
print("[probe] container root schemas:", _root.GetAppliedSchemas(), flush=True)
for _c in Usd.PrimRange(_root, Usd.TraverseInstanceProxies()):
    if _c.HasAPI(UsdPhysics.CollisionAPI):
        _en = _c.GetAttribute("physics:collisionEnabled").Get()
        print(f"[probe] collider {_c.GetPath()} type={_c.GetTypeName()} enabled={_en} approx={_c.GetAttribute('physics:approximation').Get() if _c.GetAttribute('physics:approximation') else None}", flush=True)
_cp = stage.GetPrimAtPath("/World/envs/env_0/Container/Container/Container_B06_01")
if _cp.IsValid():
    _a = _cp.GetAttribute("physics:approximation")
    print("[probe] live approximation =", _a.Get(), flush=True)
    for _sp in _a.GetPropertyStack(Usd.TimeCode.Default()):
        print("[probe]   opinion from layer:", _sp.layer.identifier, "value:", _sp.default, flush=True)
    for _ps in _cp.GetPrimStack():
        print("[probe]   prim spec layer:", _ps.layer.identifier, flush=True)
_fs = Usd.Stage.Open("/public/home/liaodl/ambench/src/ambench/source/ambench/ambench/assets/objects/container.usd", Usd.Stage.LoadNone)
_fp = _fs.GetPrimAtPath("/Root/Container/Container/Container_B06_01")
print("[probe] file (LoadNone) approximation =", _fp.GetAttribute("physics:approximation").Get() if _fp.IsValid() else "<no prim>", flush=True)
for pth in ("/World/envs/env_0/Container", "/World/envs/env_0/Ball"):
    print("[mat]", pth); print(describe_physics_material(pth), flush=True)

if args.finger_mu:
    sf, df = (float(x) for x in args.finger_mu.split(":"))
    mat_path = "/World/Physics_Materials/FingerMat"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(static_friction=sf, dynamic_friction=df, restitution=0.0))
    for side in ("l", "r"):
        sim_utils.bind_physics_material(f"/World/envs/env_0/Robot/manipulation_{side}finger_link", mat_path)
    print("[cfg] finger material bound:", args.finger_mu)
    for pth in ("/World/envs/env_0/Robot/manipulation_lfinger_link",):
        print("[mat] after bind"); print(describe_physics_material(pth), flush=True)

if args.finger_mass is not None:
    pv = env.robot.root_physx_view
    names = env.robot.body_names
    idx = [names.index("manipulation_lfinger_link"), names.index("manipulation_rfinger_link")]
    masses = pv.get_masses().clone(); inertias = pv.get_inertias().clone()
    scale = args.finger_mass / masses[0, idx[0]].item()
    masses[:, idx] = args.finger_mass; inertias[:, idx, :] *= scale
    all_idx = torch.arange(masses.shape[0], dtype=torch.int32)
    pv.set_masses(masses, all_idx); pv.set_inertias(inertias, all_idx)
    print(f"[cfg] finger mass set to {args.finger_mass} (scale {scale:.1f}); masses now {pv.get_masses()[0, idx].tolist()}")
obs, _ = env.reset()
ep = gym.spec(args.task).kwargs["scripted_policy_entry_point"]
mod, cls = ep.split(":")
Policy = getattr(importlib.import_module(mod), cls)
policy = Policy(env=env, inject_noise=False)
policy.reset()

robot = env.robot
names = robot.body_names
i_ee, i_l, i_r = names.index("ee_link"), names.index("manipulation_lfinger_link"), names.index("manipulation_rfinger_link")
gid = env.gripper_joint_ids
print("body_names:", names)
print("gripper joint ids:", gid, "names:", [robot.joint_names[i] for i in gid])
print("ball grasp offset:", env.cfg.ball_grasp_offset_local, "ball_radius:", env.cfg.ball_radius, "ball mass:", env.cfg.ball_mass)

def rel(p_w, ee_pos, ee_quat):
    return math_utils.quat_apply_inverse(ee_quat, p_w - ee_pos)

i_base = names.index("base_link")
try:
    from ambench.robots.visuals.propeller_specs import FA_HEXA_PROP_POSITIONS
    PROPS = FA_HEXA_PROP_POSITIONS.to(env.device)
    print("[cfg] prop positions (base frame):", [tuple(round(v, 3) for v in r) for r in PROPS.tolist()])
except Exception as e:
    PROPS = None; print("[cfg] no prop positions:", e)

def report(step, note=""):
    ee_pos = robot.data.body_link_pos_w[:, i_ee]; ee_quat = robot.data.body_link_quat_w[:, i_ee]
    b_pos = robot.data.body_link_pos_w[:, i_base]; b_quat = robot.data.body_link_quat_w[:, i_base]
    ball_b = rel(env.ball.data.root_pos_w, b_pos, b_quat)[0]
    dprop = ""
    if PROPS is not None:
        d = torch.norm(PROPS - ball_b.unsqueeze(0), dim=1)
        dprop = f" ball_base=({ball_b[0]:.3f},{ball_b[1]:.3f},{ball_b[2]:.3f}) nearest_prop={d.min().item():.3f}m(#{d.argmin().item()})"
    note = note + dprop
    lf = rel(robot.data.body_link_pos_w[:, i_l], ee_pos, ee_quat)[0]
    rf = rel(robot.data.body_link_pos_w[:, i_r], ee_pos, ee_quat)[0]
    ball = rel(env.ball.data.root_pos_w, ee_pos, ee_quat)[0]
    bv = env.ball.data.root_lin_vel_w[0]
    jp = robot.data.joint_pos[0, gid]
    jt = robot._data.joint_pos_target[0, gid] if hasattr(robot, "_data") else torch.zeros(2)
    fr = robot.data.applied_torque[0, gid] if hasattr(robot.data, "applied_torque") else torch.zeros(2)
    print(f"[dbg] t={step:4d} {note} grip_q={jp[0]:.4f},{jp[1]:.4f} tgt={jt[0]:.4f},{jt[1]:.4f} drive_f={fr[0]:7.2f},{fr[1]:7.2f} | "
          f"lfinger_ee=({lf[0]:.3f},{lf[1]:.3f},{lf[2]:.3f}) rfinger_ee=({rf[0]:.3f},{rf[1]:.3f},{rf[2]:.3f}) | "
          f"ball_ee=({ball[0]:.3f},{ball[1]:.3f},{ball[2]:.3f}) ball_v_w=({bv[0]:.2f},{bv[1]:.2f},{bv[2]:.2f}) ball_dz_mm={ball[2].item()*1000:6.1f} ee_z={ee_pos[0,2]:.3f}", flush=True)

report(0, "after-reset")
_land = {"done": False, "prev_z": None, "released": False}
def track_release(step):
    if _land["released"] or step < 200: return
    q = robot.data.joint_pos[0, gid].mean().item()
    if q > 0.041:  # beyond the ball-holding width (0.038): fingers are opening
        _land["released"] = True
        bv = env.ball.data.root_lin_vel_w[0]; ev = robot.data.body_link_vel_w[0, i_ee, :3]
        sol = getattr(policy, "toss_solution", {})
        print(f"[release] t={step} ball_v=({bv[0]:.2f},{bv[1]:.2f},{bv[2]:.2f}) |v|={torch.norm(bv).item():.2f} ee_v=({ev[0]:.2f},{ev[1]:.2f},{ev[2]:.2f}) |ee_v|={torch.norm(ev).item():.2f} solution v={sol.get('v',0):.2f} v_cmd={sol.get('v_cmd',0):.2f} t_R={sol.get('t_R')} rel_step={sol.get('release_step')} x_land_pred={sol.get('x_land_pred',0):.2f} aim={getattr(policy,'_aim_x',0):.2f} deadline={sol.get('deadline_hit')}", flush=True)
def track_landing(step):
    b = env.ball.data.root_pos_w[0]; z = b[2].item()
    if _land["prev_z"] is not None and not _land["done"] and _land["prev_z"] > 0.17 >= z and step > 200:
        c = env.container.data.root_pos_w[0]
        print(f"[landing] t={step} ball crosses z=0.17 at ({b[0]:.2f},{b[1]:.2f}) container=({c[0]:.2f},{c[1]:.2f}) d_xy=({b[0]-c[0]:+.2f},{b[1]-c[1]:+.2f}) in_bin={abs(b[0]-c[0])<=0.2 and abs(b[1]-c[1])<=0.15}", flush=True)
        _land["done"] = True
    _land["prev_z"] = z
    # trace the approach to the bin: any step with the ball within 0.35 m (xy) of the container and below 0.5 m
    c = env.container.data.root_pos_w[0]
    if _land.get("fine", 0) > 0 or (abs(b[0]-c[0]) < 0.6 and abs(b[1]-c[1]) < 0.6 and z < 1.0 and step > 300 and _land.get("fine_done") is None):
        if _land.get("fine", 0) == 0: _land["fine"] = 90
        _land["fine"] -= 1
        if _land["fine"] == 0: _land["fine_done"] = True
        v = env.ball.data.root_lin_vel_w[0]
        print(f"[fine] t={step} ball_rel=({b[0]-c[0]:+.3f},{b[1]-c[1]:+.3f},{z:.3f}) v=({v[0]:.2f},{v[1]:.2f},{v[2]:.2f})", flush=True)
    if abs(b[0]-c[0]) < 0.35 and abs(b[1]-c[1]) < 0.35 and z < 0.5 and step > 200:
        ok, crit = env._get_success()
        v = env.ball.data.root_lin_vel_w[0]
        print(f"[bin] t={step} ball=({b[0]-c[0]:+.2f},{b[1]-c[1]:+.2f},{z:.2f}) v=({v[0]:.1f},{v[1]:.1f},{v[2]:.1f}) crit={ {k: int(vv[0].item()) for k, vv in crit.items()} }", flush=True)
def final_report():
    ball = env.ball.data.root_pos_w[0]; c = env.container.data.root_pos_w[0]
    ok, crit = env._get_success()
    print(f"[final] ball_w=({ball[0]:.2f},{ball[1]:.2f},{ball[2]:.2f}) container_w=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}) d_xy=({ball[0]-c[0]:+.2f},{ball[1]-c[1]:+.2f}) success={ok[0].item()} crit={ {k: int(v[0].item()) for k, v in crit.items()} }", flush=True)
with torch.inference_mode():
    for step in range(1, args.steps + 1):
        if args.hold or args.motion:
            if step == 1:
                hold_action = policy.advance(obs, 0).clone()
                print("[cfg] hold action:", hold_action.tolist(), "shape", tuple(hold_action.shape), flush=True)
            action = hold_action.clone()
            if args.motion:
                f = min(step / args.ramp, 1.0) * args.amp
                if args.motion == "transx": action[0, 0] += f
                elif args.motion == "transy": action[0, 1] += f
                elif args.motion == "transz": action[0, 2] += f
                else:
                    ang = torch.tensor([f], device=action.device)
                    zero = torch.zeros_like(ang)
                    r, pch, y = (ang, zero, zero) if args.motion == "roll" else (zero, ang, zero) if args.motion == "pitch" else (zero, zero, ang)
                    q = math_utils.quat_from_euler_xyz(r, pch, y)[0]
                    action[0, 3:7] = math_utils.quat_mul(q.unsqueeze(0), hold_action[0, 3:7].unsqueeze(0))[0]
        else:
            action = policy.advance(obs, 0)
        obs, _rew, _term, _trunc, _extras = env.step(action)
        if bool(_term[0].item()):
            _sc = {k: int(v[0].item()) for k, v in _extras.get("success_criteria", {}).items()}
            print(f"[terminated] t={step} SUCCESS (episode terminated by the task's success check) criteria={_sc}", flush=True)
            break
        if bool(_trunc[0].item()):
            print(f"[truncated] t={step} timeout without success", flush=True)
            break
        track_landing(step)
        track_release(step)
        if step % args.every == 0 or step <= 3:
            report(step)
final_report()
env.close()
simulation_app.close()
