# VLM 失败理解先导实验

**问题**：在 AM-Bench 迭代中，最关键的几次诊断都来自人看回放视频（例如"夹爪一直是紧的，但框会滑脱"）。通用 VLM 能不能从同样的视频里看出这些问题，并判断出失败是策略造成的还是仿真环境造成的？

## 测试案例

| 组 | 案例 | 真值 | 来源 |
|---|---|---|---|
| 环境缺陷 | `frame_fail`、`toss_fail`、`wipe_fail` | 失败由仿真环境造成（见 `docs/仿真修复清单.md`） | 修复前的脚本专家录制 |
| 策略失败 | `push_policy`、`peg_policy`、`pull_policy` | 环境正常，失败在策略或观测 | π₀.₅ 评测回放 |
| 成功 | `frame_ok`、`toss_ok`、`wipe_ok` | 已完成任务 | 修复后的专家录制 |

每个案例 2 集。每帧由场景相机、腕部相机、机载相机横向拼接。输入有两种：12 张均匀采样的关键帧（带仿真时刻），或每 0.5 秒一帧、4 帧/秒播放的视频。

人当时看视频说出的关键现象和评分标准见 `rubric.md`。

## 条件

| 维度 | 取值 |
|---|---|
| 模型 | `deepseek-flash`（DeepSeek 官方，只支持图片）、`kimi-k3`（Moonshot 官方，图片与视频）、`qwen3.8-max`（阿里云百炼，图片与视频） |
| 提示词 | `P1_describe` 开放描述；`P2_diagnose` 诊断并区分策略与环境；`P3_checklist` 通用检查清单加鉴别诊断，要求每侧至少两个假设；`P3b_evidence` 同一清单，但不限假设数量，任何一侧无假设时必须给出排除它的画面证据，并给每个假设分配概率 |
| 重复 | 每个条件 2 次 |

提示词原文在 `run_pilot.py`。检查清单只列通用类别，不含任何与测试案例相关的例子；它是在已知答案后写的，存在偏向这三个缺陷的风险，后续应在未见过的缺陷类型上检验。

## 运行

```bash
python3 -m venv .venv && .venv/bin/pip install openai 'httpx[socks]'
# .env 中填写 DEEPSEEK_API_KEY、MOONSHOT_API_KEY、BAILIAN_API_KEY（不要提交）
.venv/bin/python run_pilot.py --dry-run                     # 只打印请求
.venv/bin/python run_pilot.py --models qwen3.8-max --prompts P3b_evidence --out raw_p3b_qwen.jsonl --resume --workers 4
```

`--resume` 会跳过输出文件里已有成功回答的条件，失败的调用会重试。Kimi 账户有组织级速率限制，并发建议不超过 3。

## 重建输入

`data/` 与 `inputs/` 下的帧和视频较大，没有放进仓库。重建方法：

1. 从 HPC 拷贝看板数据到 `data/`：修复前的专家录制在 `$A/webroot/_trash/ambench_rollouts/` 与 `_trash/ambench_data/`（`*_expert3`），修复后的在 `$A/webroot/ambench_rollouts/` 与 `viewer/ambench_data/`（`FrameAssemblyFAHexaAbsPID_fix`、`TossBallFAHexaAbsPID_ballistic`、`WipeWindowFAHexaAbsPID_servo3`）；策略失败的视频在 `$A/outputs/eval_hpc/{383328_PushSlider,385030_PegInHole,386370_PullLever}/videos/`，放到 `data/eval/`。
2. 运行 `python3 build_inputs.py`，再运行 `python3 build_eval_inputs.py`。生成的 `inputs/manifest.json` 已随仓库提交，可以对照检查。

## 结果

原始回答在 `results/*.jsonl`，每行一个条件，包含模型、输入方式、提示词、案例、真值标签和完整回答。汇总见 `RESULTS.md`。
