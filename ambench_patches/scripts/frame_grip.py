"""抓取时刻夹爪到底夹住框架没有：state=[ee(3) quat(4) gripper(1) frame(3) peg(3)]，action 最后一维=夹爪指令。"""
import glob, sys
import numpy as np, pyarrow.parquet as pq
root = sys.argv[1]
T = [900, 960, 1000, 1100, 1700, 2400, 2750, 3100]
for f in sorted(glob.glob(f"{root}/data/chunk-*/*.parquet"))[:6]:
    t = pq.read_table(f, columns=["observation.state", "action", "episode_index"])
    s = np.asarray(t.column("observation.state").to_pylist(), dtype=np.float32)
    a = np.asarray(t.column("action").to_pylist(), dtype=np.float32)
    ep = np.asarray(t.column("episode_index").to_pylist())
    for e in np.unique(ep):
        ss, aa = s[ep == e], a[ep == e]
        if len(ss) < 3200: continue
        print(f"\n=== ep{e} len={len(ss)}  gripper_obs 范围 [{ss[:,7].min():.4f},{ss[:,7].max():.4f}]  action_grip 范围 [{aa[:,-1].min():.2f},{aa[:,-1].max():.2f}]")
        print(f"state dim={ss.shape[1]}")
        print(f"{'t':>5} {'grip_obs':>9} {'act_grip':>9} {'ee x/y/z':>24}")
        for i in T:
            print(f"{i:>5} {ss[i,7]:9.4f} {aa[i,-1]:9.2f} {ss[i,0]:8.3f}{ss[i,1]:8.3f}{ss[i,2]:8.3f}")
        g = ss[1000:2750, 7]
        print(f"  闭合搬运期间(t=1000..2750) gripper_obs min={g.min():.4f} max={g.max():.4f} mean={g.mean():.4f}")
