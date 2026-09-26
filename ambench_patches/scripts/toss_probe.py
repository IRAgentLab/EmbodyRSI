"""TossBall 修复后：夹爪开度时间线 + 末端轨迹（state 只有 8 维，球位置看录制日志）。"""
import glob, sys, numpy as np, pyarrow.parquet as pq
root = sys.argv[1]
for f in sorted(glob.glob(f"{root}/data/chunk-*/*.parquet"))[:8]:
    t = pq.read_table(f, columns=["observation.state", "action", "episode_index"])
    s = np.asarray(t.column("observation.state").to_pylist(), dtype=np.float32)
    a = np.asarray(t.column("action").to_pylist(), dtype=np.float32)
    ep = np.asarray(t.column("episode_index").to_pylist())
    for e in np.unique(ep):
        ss, aa = s[ep == e], a[ep == e]
        g, ee = ss[:, 7], ss[:, 0:3]
        rel = np.where(aa[:, -1] > 0)[0]; t_rel = int(rel[0]) if len(rel) else -1
        closed = np.where(g < 0.09)[0]
        print(f"\n=== ep{e} len={len(ss)} 松爪指令 t={t_rel}")
        print("  t     grip   ee x     y     z")
        for i in [0, 5, 20, 60, 150, 300, 420, 500] + ([t_rel-2, t_rel+2, t_rel+20, t_rel+60] if t_rel > 0 else []):
            i = min(i, len(ss)-1)
            print(f"  {i:4d}  {g[i]:.4f}  {ee[i,0]:6.3f} {ee[i,1]:6.3f} {ee[i,2]:6.3f}")
        print(f"  闭合段(t=30..t_rel) 开度 min={g[30:max(t_rel,31)].min():.4f} max={g[30:max(t_rel,31)].max():.4f}")
