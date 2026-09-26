"""VLM 失败理解先导实验：让 deepseek-flash / kimi-k3 看 AM-Bench rollout，检查它们能否看出用户当时从视频里看出的问题。

用法：
  .venv/bin/python run_pilot.py --dry-run            # 只打印请求规模，不调用 API
  .venv/bin/python run_pilot.py                      # 全部条件
  .venv/bin/python run_pilot.py --models kimi-k3 --cases toss_fail --repeats 1

API key 从同目录 .env 读取（DEEPSEEK_API_KEY、MOONSHOT_API_KEY），不会打印。
结果追加写入 results/raw.jsonl，每条记录包含条件、完整回答和耗时。
"""
import argparse, base64, json, os, time
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
MANIFEST = json.load(open(ROOT / "inputs/manifest.json"))

def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

TASKS = {
    "frame": "无人机（六旋翼，带一条机械臂和二指夹爪）需要从地面拾起一个木质方框，飞到墙边，把方框挂到墙上的四根销钉上。",
    "toss": "无人机（六旋翼，带一条机械臂和二指夹爪）开始时夹爪里握着一个红色小球，需要把球抛进前方地面上的收纳箱。规则要求机体在箱子后方至少 0.7 米处释放，即必须抛而不是飞过去放。",
    "wipe": "无人机（六旋翼，带一条机械臂和二指夹爪）开始时夹爪里握着一块黄色海绵，需要飞到窗前，用海绵依次按压并擦掉窗面上的红色污点。",
    "push": "无人机（六旋翼，带一条机械臂和二指夹爪）需要飞到墙前，抓住墙上导轨里的红色滑块，并把滑块沿导轨向右推到位。",
    "peg": "无人机（六旋翼，带一条机械臂和二指夹爪）开始时夹爪里握着一根销钉，需要飞到墙前，把销钉插进墙上的方孔。",
    "pull": "无人机（六旋翼，带一条机械臂和二指夹爪）需要飞到墙前，抓住墙上拉杆的把手，把拉杆向上拉起。",
}
VIEWS = "每帧画面由三个视角横向拼接：左边是固定的场景相机，中间是机械臂末端的腕部相机，右边是机身上朝前的机载相机。"

P3_TEXT = """{task}
{views}
你是负责排查仿真实验结果的工程师。这段 rollout 来自一个机器人仿真基准。任务失败既可能源于机器人的策略或动作，也可能源于仿真环境本身：任务设置、初始状态、物体资产、几何与碰撞、物理参数、成功判定逻辑、相机与传感器。仿真环境并不保证正确，请不要默认环境没有问题，也不要默认一定是环境的问题。

请按以下步骤作答：
1. 判断这次尝试是否完成了任务。如果完成了，说明依据，并指出过程中是否仍有任何可疑之处，然后结束。
2. 如果没有完成，按时间顺序描述实际发生了什么，重点描述机器人与每个被操作物体之间的关系如何随时间变化，以及失败大约出现在什么时刻。
3. 逐项检查下列方面，每项给出"正常 / 可疑 / 无法判断"和画面依据：
   a. 初始状态：开始时各物体的位置和状态是否与任务描述一致、是否合理；
   b. 运动与接触：物体的运动是否与它受到的接触和约束相符；
   c. 机器人自身的状态与被操作物体的状态之间是否一致；
   d. 几何与碰撞：是否有穿透，是否有本应阻挡却没有阻挡、或本应可以通过却被阻挡的情况；
   e. 物理属性：重力、质量、摩擦、弹性等表现是否合理；
   f. 机器人的动作：轨迹、时序、对准、力度是否合理；
   g. 观测：相机视角是否被遮挡，关键信息是否可见。
4. 鉴别诊断：分别列出至少两个"策略或动作问题"的假设和至少两个"仿真环境问题"的假设。每个假设写出画面中的支持证据、反对证据，以及用什么额外检查或实验可以区分它。
5. 结论：最可能的根本原因属于哪一侧（策略或动作 / 仿真环境），置信度（高 / 中 / 低），以及还缺哪些信息才能确认。"""

P3B_TEXT = P3_TEXT.replace(
    """4. 鉴别诊断：分别列出至少两个"策略或动作问题"的假设和至少两个"仿真环境问题"的假设。每个假设写出画面中的支持证据、反对证据，以及用什么额外检查或实验可以区分它。
5. 结论：最可能的根本原因属于哪一侧（策略或动作 / 仿真环境），置信度（高 / 中 / 低），以及还缺哪些信息才能确认。""",
    """4. 鉴别诊断：列出所有与画面证据相符的假设。策略或动作一侧与仿真环境一侧都要考虑，但任何一侧都可以没有假设；如果某一侧没有提出假设，必须写出排除这一侧所依据的具体画面证据，"没有看到明显异常"不算证据。每个假设写出画面中的支持证据、反对证据，以及用什么额外检查或实验可以区分它，并给出可能性估计，所有假设的可能性之和为 100%。
5. 结论：最可能的根本原因属于哪一侧（策略或动作 / 仿真环境），置信度（高 / 中 / 低），以及还缺哪些信息才能确认。""")
assert P3B_TEXT != P3_TEXT

PROMPTS = {
    "P1_describe": "{task}\n{views}\n请按时间顺序描述这段过程中实际发生了什么。重点关注机器人和被操作物体之间的关系，指出任何看起来异常或不符合物理常识的地方。",
    "P2_diagnose": ("{task}\n{views}\n请回答：\n1. 这次尝试是否完成了任务？\n2. 如果没有完成，失败的直接现象是什么，大约发生在什么时刻？\n"
                    "3. 最可能的根本原因是什么？请明确区分：是机器人的动作或策略有问题，还是仿真环境本身（物体、碰撞、物理设置）有问题。\n"
                    "4. 你的判断依据是画面中的哪些细节？"),
    "P3_checklist": P3_TEXT,
    "P3b_evidence": P3B_TEXT,
}

OPENROUTER = "https://openrouter.ai/api/v1"
MODELS = {  # name -> provider config; "id" is the provider's model id
    "deepseek-v4.1-flash": {"base_url": OPENROUTER, "key": "OPENROUTER_API_KEY", "id": "deepseek/deepseek-v4.1-flash", "modes": ["frames"], "video": "data"},
    "kimi-k3": {"base_url": OPENROUTER, "key": "OPENROUTER_API_KEY", "id": "moonshotai/kimi-k3", "modes": ["video", "frames"], "video": "data"},
    # 官方通道（可选）：
    "deepseek-flash-official": {"base_url": "https://api.deepseek.com", "key": "DEEPSEEK_API_KEY", "id": "deepseek-flash", "modes": ["frames"], "video": None},
    "qwen3.8-max": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "key": "BAILIAN_API_KEY", "id": "qwen3.8-max", "modes": ["video", "frames"], "video": "data"},
    "kimi-k3-official": {"base_url": "https://api.moonshot.cn/v1", "key": "MOONSHOT_API_KEY", "id": "kimi-k3", "modes": ["video", "frames"], "video": "upload"},
}
DEFAULT_MODELS = ["deepseek-flash-official", "kimi-k3-official"]

def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()

_uploaded = {}
def video_part(client, path, how):
    if how == "data":
        return {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + b64(path)}}
    if path not in _uploaded:
        fo = client.files.create(file=Path(path), purpose="video")
        _uploaded[path] = fo.id
    return {"type": "video_url", "video_url": {"url": f"ms://{_uploaded[path]}"}}

def build_content(client, item, mode, prompt_text, dry, how=None):
    parts = []
    if mode == "video":
        parts.append({"type": "video_url", "video_url": {"url": "ms://<dry-run>"}} if dry else video_part(client, item["video"], how))
        text = prompt_text + f"\n（视频共约 {item['duration_s']} 秒，按 4 帧/秒播放，每帧对应仿真中 0.5 秒。）"
    else:
        for p in item["keyframes"]:
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{'<b64>' if dry else b64(p)}"}})
        ts = "、".join(f"{t}s" for t in item["keyframe_t"])
        text = prompt_text + f"\n（以上 {len(item['keyframes'])} 张图按时间顺序排列，对应的仿真时刻依次为：{ts}。）"
    parts.append({"type": "text", "text": text})
    return parts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--prompts", nargs="*", default=list(PROMPTS))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="raw.jsonl", help="results/ 下的输出文件名")
    ap.add_argument("--modes", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true", help="跳过 raw.jsonl 里已有成功回答的条件")
    a = ap.parse_args()
    load_env()
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    import threading
    from concurrent.futures import ThreadPoolExecutor
    fout = open(out / a.out, "a"); lock = threading.Lock()
    done = set()
    if a.resume and (out / a.out).exists():
        for line in open(out / a.out):
            r = json.loads(line)
            if r.get("answer"):
                done.add((r["model"], r["mode"], r["prompt"], r["case"], r["demo"], r["rep"]))
    jobs = []
    for mname in a.models:
        cfg = MODELS[mname]
        key = os.environ.get(cfg["key"])
        if not key and not a.dry_run:
            print(f"!! 缺少 {cfg['key']}，跳过 {mname}"); continue
        client = None if a.dry_run else OpenAI(api_key=key, base_url=cfg["base_url"], timeout=900)
        for item in MANIFEST:
            if a.cases and item["case"] not in a.cases: continue
            task = TASKS[item["case"].split("_")[0]]
            for mode in cfg["modes"]:
                if a.modes and mode not in a.modes: continue
                for pid in a.prompts:
                    for r in range(a.repeats):
                        if (mname, mode, pid, item["case"], item["demo"], r) in done: continue
                        jobs.append((mname, cfg, client, item, task, mode, pid, r))
    n = len(jobs)
    def work(job):
        mname, cfg, client, item, task, mode, pid, r = job
        ptext = PROMPTS[pid].format(task=task, views=VIEWS)
        tag = f"{mname} | {mode} | {pid} | {item['case']}/{item['demo']} | rep{r}"
        if a.dry_run:
            print(f"[dry] {tag}"); return
        t0 = time.time()
        try:
            content = build_content(client, item, mode, ptext, False, cfg["video"])
            resp = client.chat.completions.create(model=cfg["id"], messages=[{"role": "user", "content": content}])
            ans = resp.choices[0].message.content
            usage = resp.usage.model_dump() if resp.usage else None
            err = None
        except Exception as e:
            ans, usage, err = None, None, f"{type(e).__name__}: {e}"
        rec = {"model": mname, "mode": mode, "prompt": pid, "case": item["case"], "demo": item["demo"],
               "label": item["label"], "truth": item.get("truth"), "rep": r, "answer": ans, "error": err, "usage": usage,
               "latency_s": round(time.time() - t0, 1), "time": time.strftime("%F %T")}
        with lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            print(f"{tag}  {'ERR ' + err[:120] if err else 'ok'}  {rec['latency_s']}s", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, jobs))
    print(f"=== {n} 个请求")

if __name__ == "__main__":
    main()
