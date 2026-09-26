"""汇总评分：按 真值组 × 模型 × 输入方式 × 提示词 统计。输出 Markdown 表到 stdout 和 summary.json。"""
import glob, json, collections, os
HERE = os.path.dirname(os.path.abspath(__file__))
items = {}
for f in glob.glob(f"{HERE}/chunk_*.json"):
    for it in json.load(open(f)):
        items[it["id"]] = it
scores = {}
for f in glob.glob(f"{HERE}/scores_*.jsonl"):
    for line in open(f):
        line = line.strip()
        if line:
            s = json.loads(line); scores[s["id"]] = s
rows = []
for i, it in items.items():
    if i in scores:
        rows.append({**{k: it[k] for k in ("case", "truth", "model", "mode", "prompt")}, **scores[i]})
missing = len(items) - len(rows)

MODEL = {"deepseek-flash-official": "DeepSeek", "kimi-k3-official": "Kimi K3", "qwen3.8-max": "Qwen3.8-Max"}
PROMPTS = ["P1_describe", "P2_diagnose", "P3_checklist", "P3b_evidence"]
def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)
def pct(v): return "—" if v is None else f"{v*100:.0f}%"
def num(v): return "—" if v is None else f"{v:.2f}"

def table_by(key_fn, header, groups):
    out = []
    for truth, cols in groups:
        sub = [r for r in rows if r["truth"] == truth]
        keys = sorted({key_fn(r) for r in sub}, key=lambda k: [PROMPTS.index(x) if x in PROMPTS else 9 for x in (k if isinstance(k, tuple) else (k,))] + [str(k)])
        out.append(f"\n**{header[truth]}**\n")
        out.append("| 条件 | n | " + " | ".join(c[0] for c in cols) + " |")
        out.append("|---|---|" + "---|" * len(cols))
        for k in keys:
            g = [r for r in sub if key_fn(r) == k]
            cells = [c[1](g) for c in cols]
            name = " / ".join(MODEL.get(x, x) for x in (k if isinstance(k, tuple) else (k,)))
            out.append(f"| {name} | {len(g)} | " + " | ".join(cells) + " |")
    return "\n".join(out)

env_cols = [("关键现象 K", lambda g: num(mean([r["K"] for r in g])[0])),
            ("归到环境侧", lambda g: pct(mean([1 if r["side"] == "env" else 0 for r in g])[0])),
            ("含 mixed", lambda g: pct(mean([1 if r["side"] in ("env", "mixed") else 0 for r in g])[0])),
            ("根因 R", lambda g: num(mean([r["R"] for r in g])[0])),
            ("幻觉", lambda g: pct(mean([r["halluc"] for r in g])[0]))]
pol_cols = [("关键现象 K", lambda g: num(mean([r["K"] for r in g])[0])),
            ("正确归到策略侧", lambda g: pct(mean([1 if r["side"] == "policy" else 0 for r in g])[0])),
            ("误判为环境", lambda g: pct(mean([1 if r["side"] == "env" else 0 for r in g])[0])),
            ("mixed", lambda g: pct(mean([1 if r["side"] == "mixed" else 0 for r in g])[0])),
            ("幻觉", lambda g: pct(mean([r["halluc"] for r in g])[0]))]
ok_cols = [("判为成功", lambda g: pct(mean([1 if r["said_success"] == "yes" else 0 for r in g])[0])),
           ("判为失败", lambda g: pct(mean([1 if r["said_success"] == "no" else 0 for r in g])[0])),
           ("幻觉", lambda g: pct(mean([r["halluc"] for r in g])[0]))]
header = {"env": "环境缺陷组（frame_fail、toss_fail、wipe_fail）", "policy": "策略失败组（push、peg、pull）", "ok": "成功组"}
groups = [("env", env_cols), ("policy", pol_cols), ("ok", ok_cols)]

print(f"评分条数 {len(rows)} / {len(items)}（缺 {missing}）")
print("\n### 按提示词")
print(table_by(lambda r: r["prompt"], header, groups))
print("\n### 按模型 × 提示词")
print(table_by(lambda r: (r["model"], r["prompt"]), header, groups))
print("\n### 按模型 × 输入方式（全部提示词合并）")
print(table_by(lambda r: (r["model"], r["mode"]), header, groups))
print("\n### 按案例 × 提示词（环境缺陷组）")
env_rows = [r for r in rows if r["truth"] == "env"]
for case in ("frame_fail", "toss_fail", "wipe_fail"):
    print(f"\n**{case}**\n\n| 提示词 | n | K | 归到环境侧 | R |\n|---|---|---|---|---|")
    for p in PROMPTS:
        g = [r for r in env_rows if r["case"] == case and r["prompt"] == p]
        if g:
            print(f"| {p} | {len(g)} | {num(mean([r['K'] for r in g])[0])} | {pct(mean([1 if r['side']=='env' else 0 for r in g])[0])} | {num(mean([r['R'] for r in g])[0])} |")
json.dump(rows, open(f"{HERE}/scored_rows.json", "w"), ensure_ascii=False)
