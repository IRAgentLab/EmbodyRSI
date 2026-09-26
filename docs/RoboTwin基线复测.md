# RoboTwin 2.0 上的 π₀.₅ 基线：现状与复测清单

2026-09-26 核对。来源：RoboTwin 仓库 main 分支（commit `ea8b211`，2026-09-24）、XPolicyLab（commit `d6332bf`，2026-09-24）、官方排行榜数据 `https://robotwin-platform.github.io/data/robotwin_leaderboard.json`（2026-09-24 更新）。

## 结论

- **π₀.₅ 已有官方数字**：50 任务联合训练（co-train），clean2clean（easy）70.7，clean2random（hard）46.0。之前"RoboTwin 2.0 没有 π₀.₅ 数字、需自行补测"的说法作废。
- **官方 checkpoint 已放出**：`TianxingChen/RoboTwin2.0` 数据集仓库下 `cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz`，约 12.4 GB。
- 因此基线工作从"训练"变为"**下载官方 checkpoint，在我们的集群上复测，确认环境能复现 70.7 / 46.0**"。之后所有对照组都以它为起点或参照。

## 标准设定

| 项 | 内容 |
|---|---|
| 本体 | Aloha-AgileX 双臂，三路相机：头部 `cam_high`、左右腕部 `cam_left_wrist` / `cam_right_wrist`，默认 240×320 |
| 训练数据 | 50 任务 × 每任务 50 条 `demo_clean`，共 2,500 条 |
| easy（clean2clean） | 在 `demo_clean` 配置下评测：无域随机化，`eval_instruction: seen` |
| hard（clean2random） | 在 `demo_randomized` 配置下评测：随机背景、杂乱桌面、随机光照、桌高 ±3 cm，且 `eval_instruction: unseen`（**同时换了没见过的语言指令**） |
| 评测次数 | 每任务 100 次（`test_num` 默认 100） |
| 种子筛选 | 评测前先让脚本专家在该种子上跑一遍，专家失败或场景不稳定的种子跳过（`expert_check` 默认开启），种子从 `100000 × (1 + seed)` 开始 |
| 步数上限 | 按任务设定，400–1500 步，见 `env_cfg/task_config/_eval_step_limit.yml` |
| 排行榜两条赛道 | co-train（一个模型联合训练 50 任务）与 single-task SFT（每任务一个 checkpoint） |

注意 hard 同时改变了视觉和语言两个因素，诊断 hard 下的失败时要把两者分开。

## 测试侧隔离：哪些因素在训练与评测间分开

按代码核对（`envs/_base_task.py`、`envs/utils/rand_create_cluttered_actor.py`、`description/`）：

| 因素 | 训练与评测是否分开 | 依据 |
|---|---|---|
| 背景与桌面纹理 | **分开**：录制用 `seen` 纹理库，评测用 `unseen` 纹理库 | `_base_task.py:278`，按 `eval_mode` 选 `assets/background_texture/{seen,unseen}` |
| 任务指令模板 | **分开**：每任务 50 条 seen、10 条 unseen | `description/task_instruction/<task>.json` 的 `seen` / `unseen` 字段 |
| 指令里的物体描述 | **分开**：例如 `001_bottle/base8.json` 有 12 条 seen、3 条 unseen | `description/objects_description/<obj>/base*.json` |
| 桌面杂物 | **不分开**：从同一个杂物物体列表里随机取，只排除当前任务用到的物体 | `get_available_cluttered_objects()` 不看 `eval_mode` |
| 光照 | **不分开**：方向光与点光颜色直接 `np.random.rand()` | `_base_task.py` 约 236–262 行 |
| 桌高 | **不分开**：`random_table_height` 范围内均匀采样 | `_base_task.py:112` |

`eval_mode` 在 `_base_task.py` 里只影响两处：纹理库选择（278 行）和读取步数上限（142 行）。

**含义**：在纹理和语言上，即使把 `demo_randomized` 加进训练，测的仍是泛化；在杂物、光照、桌高上，加随机化数据就是在覆盖测试分布。我们的协议需要自行切分这三个因素，见 `研究规划.md` 第 7 节。

**Agent 可见性**：`unseen` 指令模板、`unseen` 物体描述和 `unseen` 纹理目录都不能出现在 Agent 的工作区里，hard 评测的视频与记录也不能给 Agent 看。Agent 可以知道"评测会换没见过的说法"并自己生成改写，但要统计改写与 unseen 集的重合度并报告。

## 工具链（2026-08 起统一到 XPolicyLab）

旧版仓库里的 `policy/pi0` 等目录已移除，训练与部署统一走 XPolicyLab（RoboTwin 以 git submodule 引入）。

| 步骤 | 命令或位置 |
|---|---|
| 下载演示 | `bash scripts/download_xpolicylab_data.sh [task ...]`，落到 `data/demo_clean/<task>/aloha_agilex/data/` |
| 自己录制 | `bash collect_data.sh <task> demo_clean\|demo_randomized <gpu>`：先搜索专家能成功的种子，再回放录制；输出 HDF5 |
| 转 LeRobot | `python XPolicyLab/scripts/transform_lerobot_v21_format.py "demo_clean.*.aloha_agilex" --repo_id ... --max_episode 50`（v3.0 用 `transform_lerobot_v30_format.py`） |
| π₀.₅ 适配器 | `XPolicyLab/policy/Pi_05/`：`install.sh`、`process_data.sh`、`train.sh`、`eval.sh`、`deploy.yml`；openpi 由 uv 管理，不用 conda |
| 默认训练配置 | `pi05_base_aloha_full_sim_arx-x5_seed_0`：从 `pi05_base` 起，batch 256，60k 步，FSDP 2 卡。**这是为 RoboDojo 写的配置**，官方 RoboTwin checkpoint 是否用同一组超参未确认 |
| 多任务评测 | `bash scripts/eval_policy.sh multitask --config env_cfg/eval/all_tasks.yml --policy-name Pi_05 --ckpt-name ... --env-cfg-type arx_x5 ...`，按 GPU 池给每个任务起策略服务和仿真 |

**图像解码的坑**：HDF5 里的图像有两种字节格式（旧数据通道是反的），必须用 `decode_image_bit` 解码，不要自己用 `cv2.imdecode` 再加 `BGR2RGB`。自己写数据加载器或改录制时尤其要注意。

## 官方 π₀.₅ 逐任务成绩（co-train，按 hard 升序）

π₀ 一列来自 2025-08 的单任务 SFT 赛道，训练设定不同，只作参考，不能与 π₀.₅ 直接比。

| 任务 | π₀.₅ easy | π₀.₅ hard | easy − hard | π₀ easy | π₀ hard |
|---|---|---|---|---|---|
| Handover Mic | 98 | 6 | 92 | 98 | 13 |
| Move Can Pot | 60 | 10 | 50 | 58 | 21 |
| Place Can Basket | 56 | 10 | 46 | 41 | 5 |
| Handover Block | 30 | 12 | 18 | 45 | 8 |
| Hanging Mug | 17 | 14 | 3 | 11 | 3 |
| Move Stapler Pad | 20 | 18 | 2 | 0 | 2 |
| Blocks Ranking Size | 45 | 21 | 24 | 7 | 1 |
| Rotate QRcode | 90 | 21 | 69 | 68 | 15 |
| Stamp Seal | 53 | 21 | 32 | 3 | 4 |
| Put Object Cabinet | 39 | 22 | 17 | 68 | 18 |
| Beat Block Hammer | 93 | 23 | 70 | 43 | 21 |
| Stack Blocks Three | 77 | 25 | 52 | 17 | 0 |
| Turn Switch | 52 | 25 | 27 | 27 | 23 |
| Place Fan | 66 | 26 | 40 | 20 | 10 |
| Place Object Basket | 70 | 29 | 41 | 16 | 2 |
| Place Mouse Pad | 34 | 31 | 3 | 7 | 1 |
| Move Pillbottle Pad | 46 | 32 | 14 | 21 | 1 |
| Click Bell | 31 | 35 | −4 | 44 | 3 |
| Lift Pot | 98 | 35 | 63 | 84 | 36 |
| Place Phone Stand | 65 | 37 | 28 | 35 | 7 |
| Place Object Scale | 78 | 38 | 40 | 10 | 0 |
| Place A2B Right | 68 | 39 | 29 | 27 | 6 |
| Scan Object | 49 | 40 | 9 | 18 | 1 |
| Pick Diverse Bottles | 66 | 42 | 24 | 27 | 6 |
| Stack Blocks Two | 91 | 43 | 48 | 42 | 1 |
| Blocks Ranking RGB | 72 | 44 | 28 | 19 | 5 |
| Place Dual Shoes | 70 | 45 | 25 | 15 | 0 |
| Place Bread Skillet | 76 | 47 | 29 | 23 | 1 |
| Place Object Stand | 85 | 47 | 38 | 36 | 11 |
| Place A2B Left | 71 | 49 | 22 | 31 | 1 |
| Click Alarmclock | 65 | 50 | 15 | 63 | 11 |
| Place Shoe | 82 | 50 | 32 | 28 | 6 |
| Put Bottles Dustbin | 78 | 50 | 28 | 54 | 13 |
| Stack Bowls Three | 81 | 52 | 29 | 66 | 24 |
| Pick Dual Bottles | 72 | 57 | 15 | 57 | 12 |
| Open Microwave | 86 | 59 | 27 | 80 | 50 |
| Place Bread Basket | 63 | 60 | 3 | 17 | 4 |
| Press Stapler | 79 | 63 | 16 | 62 | 29 |
| Move Playingcard Away | 84 | 65 | 19 | 53 | 22 |
| Open Laptop | 93 | 68 | 25 | 85 | 46 |
| Stack Bowls Two | 94 | 70 | 24 | 91 | 41 |
| Place Container Plate | 92 | 72 | 20 | 88 | 45 |
| Place Cans Plasticbox | 31 | 73 | −42 | 34 | 2 |
| Adjust Bottle | 99 | 77 | 22 | 90 | 56 |
| Place Empty Cup | 96 | 80 | 16 | 37 | 11 |
| Dump Bin Bigbin | 92 | 84 | 8 | 83 | 24 |
| Place Burger Fries | 82 | 86 | −4 | 80 | 4 |
| Grab Roller | 99 | 97 | 2 | 96 | 80 |
| Shake Bottle Horizontally | 99 | 100 | −1 | 99 | 51 |
| Shake Bottle | 100 | 100 | 0 | 97 | 60 |
| **均值** | **70.7** | **46.0** | | 46.4 | 16.3 |

几个值得诊断的点：

- **视觉或语言迁移失败**：easy 高而 hard 低，例如 Handover Mic（98 → 6）、Rotate QRcode（90 → 21）、Beat Block Hammer（93 → 23）、Lift Pot（98 → 35）。要分清是随机化背景和光照，还是没见过的指令造成的。
- **两种设定都低**：Hanging Mug（17 / 14）、Move Stapler Pad（20 / 18）、Place Mouse Pad（34 / 31）。这些更像能力缺口，也可能有环境或判据问题。
- **hard 反而更高**：Place Cans Plasticbox 在 easy 下只有 31，在 hard 下却有 73。这不符合常理，值得先查是不是 clean 配置或判据有问题。它可以作为诊断基准里"环境缺陷候选"的真实案例。

## 公榜规则与各方法实际用数据的情况

**正式规则**（排行榜页面，2026-09-26 查）只有两条：

> Training data is fixed to 50 demo_clean trajectories × 50 tasks (2,500 demos total) on Aloha-AgileX, then evaluated 100 trials/task under demo_clean and demo_randomized.

> To be listed on this leaderboard, a model must provide publicly released code, publicly released weights, and a technical report (arXiv paper or equivalent public document) describing the method.

页面和 `robotwin_leaderboard.json` 都没有提到数据增强、预训练数据、额外观测模态，既不允许也不禁止。

**两个上榜方法的实际做法**：

| 方法 | hard | 后训练数据 | 规则之外的数据使用 | 来源 |
|---|---|---|---|---|
| GigaBrain-0.7 | 67.9 | 50 条 clean × 50 任务（"post-trained jointly on all 50 benchmark tasks using 50 clean demonstrations per task"） | **预训练语料包含 RoboTwin 2.0 轨迹**："1453.92 hours of trajectories collected from configurable physics-based environments, including high-quality trajectories curated from [15, …]"，[15] 即 RoboTwin 2.0。选了哪些轨迹、是否含随机化数据，论文未说明。其 hard（67.9）与 easy（66.8）几乎持平，其他方法普遍掉 20–30 点 | arXiv 2608.15875，第 9 页与表 9 |
| ME-Dex-1.0 | 68.12 | 上榜结果只用 2,500 条 clean 演示 | **重放同一批轨迹补录触觉**："Replaying existing RoboTwin … actions … allow force sensors in the simulation to directly record tactile signals for those trajectories"；评测时触觉输入置零。另有用 27,500 条 clean 加 random 轨迹训练的版本，不是上榜结果 | arXiv 2609.21449 |
| π₀.₅ | 46.0 | 同上 | 用 `pi05_base` 起步，其预训练数据不公开，无法确认是否含 RoboTwin | XPolicyLab `Pi_05` 训练配置 |

**结论**：

1. 公榜固定的只是**后训练用的 2,500 条轨迹**。预训练不受约束；对同一批轨迹重放补录新模态已有上榜先例。
2. 所以公榜不是受控比较，hard 一栏混合了模型的泛化能力和预训练对 RoboTwin 场景的覆盖。引用榜上成绩时，要注明各方法的预训练是否含 RoboTwin（至少 GigaBrain-0.7 含）。
3. 对我们的方法：只改训练代码、以及给同一批轨迹补观测模态，这两档在公榜规则内；重写专家、补录新场景、修资产不在规则内。主评测因此用自定的等预算协议，公榜只作参照，见 `研究规划.md` 第 7、8 节。

## 排行榜参照（co-train 赛道，2026-09-24）

| 方法 | easy | hard |
|---|---|---|
| ME-Dex-1.0 | 89.58 | 68.12 |
| GigaBrain-0.7 | 66.8 | 67.9 |
| OLA-Sem | 75.12 | 67.56 |
| WorldScape Policy 2.0 | 84.92 | 38.5 |
| **π₀.₅** | **70.7** | **46.0** |
| X-VLA | 68.0 | 20.9 |

hard 的当前最好成绩约 68，但如上所述，前两名分别用了含 RoboTwin 的预训练和补录模态，与 π₀.₅ 的 46.0 口径不同。我们的主目标不在公榜上，而是在自定协议的留出测试集上比自测的 clean 训练 π₀.₅ 基线 +15～20 点。

## 复测清单

1. **环境**：确认 HPC 节点能跑 SAPIEN 的 Vulkan 渲染。先单任务、单种子试跑 `EVAL_ENV_TYPE=debug`，再试 `sim`。
2. **装环境**：`git clone --recurse-submodules` RoboTwin；执行 RoboTwin 的安装和资产下载；在 `XPolicyLab/policy/Pi_05` 下执行 `install.sh`（uv）。
3. **下 checkpoint**：下载并解压 `cotrain_ckpt/Pi_05/...tar.gz`。**先看里面的训练配置**，记下步数、batch、归一化统计量和动作表示（名字里是 `joint`），补进本文档。
4. **冒烟测试**：挑 3 个任务（一个 easy 高 hard 低、一个两者都低、一个两者都高），每任务 easy 与 hard 各 20 次，确认成功率大致对得上。
5. **全量复测**：50 任务 × 100 次 × easy / hard，共 1 万回合，用 `eval_policy.sh multitask` 在 8 卡上调度。每个任务记录与官方数字的差。
6. **判定标准**：宏平均与官方相差 3 个点以内即视为复现。个别任务偏差大的，先查种子、步数上限和指令集是否一致。
7. **留存评测记录**：打开 `eval_video_log`，保存每回合的视频与状态。这些记录就是诊断实验和 Agent 的输入。

## 仍待确认

- 官方 π₀.₅ checkpoint 的训练步数、batch size，以及数据是否只用了 `demo_clean` 50 条 × 50 任务。
- 排行榜 π₀.₅ 成绩是否通过 XPolicyLab 标准接口复现（榜单说明写的是"results here are reproduced via the XPolicyLab standard interface"，contributor 为 RoboTwin Team）。
- 单任务 SFT 赛道没有 π₀.₅ 数字。如果我们要做单任务实验，需要自己跑。
- GigaBrain-0.7 预训练里具体用了哪些 RoboTwin 轨迹、是否包括随机化数据。
- 公榜维护者对数据增强、补录模态、预训练含 RoboTwin 的态度。页面没写，只能问维护者（chentianxing2002@gmail.com）。
- 任务物体本身（不是杂物）的实例是否有 seen / unseen 切分：本次只查了纹理、指令、杂物、光照、桌高。
