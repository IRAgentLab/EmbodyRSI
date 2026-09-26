"""Apply the grasp fixes: FA-Hexa width converter + finger drive stiffness, frame/ball masses."""
import pathlib, re, shutil
R = pathlib.Path("/public/home/liaodl/ambench/src/ambench/source/ambench/ambench")

def patch(rel, pairs):
    p = R / rel
    s = p.read_text()
    bak = p.with_suffix(p.suffix + ".bak_grasp")
    if not bak.exists():
        shutil.copy(p, bak)
    for old, new, count in pairs:
        n = s.count(old)
        assert n == count, f"{rel}: expected {count} of {old!r}, found {n}"
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", rel)

# 1. FA-Hexa: width converter (same as UA-Quad) + stiffer finger drive
patch("robots/fa_hexa.py", [
    ('PROPELLER_VIZ_RELATIVE_PATHS = tuple(',
     'def gripper_joint_pos_from_object_width(object_width: float) -> float:\n'
     '    """Return the symmetric finger position that fits an object at the grasp center."""\n'
     '    closed_inner_overlap = 0.006\n'
     '    return 0.5 * (object_width + closed_inner_overlap)\n\n\n'
     'PROPELLER_VIZ_RELATIVE_PATHS = tuple(', 1),
    ('        joint_names=("arm_lfinger_joint", "arm_rfinger_joint"),\n    )',
     '        joint_names=("arm_lfinger_joint", "arm_rfinger_joint"),\n'
     '        joint_position_from_object_width=gripper_joint_pos_from_object_width,\n    )', 1),
    ('            stiffness=100.0,\n            damping=20.0,',
     '            stiffness=1000.0,\n            damping=30.0,', 2),
])
# 2. masses
patch("tasks/frame_assembly/frame_assembly_env_cfg.py", [("    frame_mass = 0.01\n", "    frame_mass = 0.3\n", 1)])
patch("tasks/toss_ball/toss_ball_env_cfg.py", [("    ball_mass = 0.01\n", "    ball_mass = 0.05\n", 1)])
import py_compile
for rel in ["robots/fa_hexa.py", "tasks/frame_assembly/frame_assembly_env_cfg.py", "tasks/toss_ball/toss_ball_env_cfg.py"]:
    py_compile.compile(str(R / rel), doraise=True)
print("compile ok")
