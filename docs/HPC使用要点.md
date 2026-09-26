# HPC 使用要点（华南理工 HPC）

通用使用方法见 [SCUTHPCSkill](https://github.com/IRAgentLab/SCUTHPCSkill)，它也可以作为 Claude Code 等编码 Agent 的技能直接安装。本页只记录这个项目踩过的坑和约定。

## 目录约定

项目根目录记为 `$A=/public/home/liaodl/ambench`。

| 路径 | 内容 |
|---|---|
| `$A/src/ambench` | AM-Bench 源码（上游 `ambench/ambench` 加本项目改动，见 `ambench_patches/`） |
| `$A/src/ambench/ext/openpi` | openpi；训练配置在 `src/openpi/training/config.py` |
| `$A/env/` | 作业脚本与辅助脚本（仓库里的 `ambench_patches/scripts/` 是其中的精选副本） |
| `$A/data/lerobot_hpc/<Task>/` | 原始录制会话，每个会话一个 `demo-<时间戳>/lerobot` |
| `$A/lerobot_home/am_bench/` | 20 Hz openpi 训练集（`st2cam_*`、`clean_*`、`mt12_*` 等） |
| `$A/outputs/eval_hpc/<作业号>_<任务>/` | 评测输出，含 `results.txt`、`tracking/`、部分有 `videos/` |
| `$A/src/ambench/ext/openpi/checkpoints/` | openpi checkpoint，每个 step 目录约 42 GB |
| `$A/webroot/` | 回放看板的数据 |
| `$A/logs/` | 所有作业日志 |

## 必须遵守的规则

1. **登录节点不跑计算**。分析、转换、导出走 CPU 分区作业（`cpu_run.sbatch`），仿真和训练走 GPU 作业。集群是多人共用的。
2. **所有 sbatch 必须带 `#SBATCH --propagate=NONE`**。否则登录节点的 `ulimit -u 1000` 会传给作业，同节点多开几个 Isaac 作业就会因为线程数超限卡死或崩溃（报错 `fork: Resource temporarily unavailable`）。`#SBATCH` 行必须写在第一条命令之前。
3. **仿真作业每次只申请 1 张 GPU**；训练按需申请，40k 步的多任务训练 8 卡约 8.5 小时。
4. **线程数按申请的核数算**，不要按节点核数开。脚本里的做法：`OMP=max(4, NT/2)`，Isaac 线程池 `KIT=max(8, NT*2/3)`，低于 8 会卡死。
5. **gpu2 节点要排除**（`--exclude=gpu2`）：驱动版本不匹配 Isaac，且连不上校园代理。坏节点记录在 `$A/logs/bad_isaac_nodes.txt`。
6. **`$A/hfcache/parquet` 是训练时的活数据**，有 openpi 训练在跑时不能删，否则训练会直接失败。
7. **计算节点上网要走代理 `login5:3128`**（`hpc_env.sh` 已设置）。CPU 分区连不上代理，所以 openpi 的归一化统计要在 GPU 训练作业里算（`train_mt.sbatch` 已这样做，并在代理不通时自动重试）。
8. **磁盘配额 4 TB**。checkpoint 最占空间；清理用 `prune3.sh`，先不加参数看清单，再 `DO=1` 执行。
9. **Isaac 偶发段错误**（RC=139）。录制中途崩溃时，已完成的集数一般完好，但最后一个文件可能写坏，导出前用 `check_parquet.py` 检查，写坏的会话用 `repair_session.py` 隔离坏文件。
10. **TossBall 录制用 6 个并行环境**，16 个会卡死在第 600 步。

## 常用作业

| 目的 | 命令示例 |
|---|---|
| 录制专家演示 | `CAMS="ee_camera base_camera" WRENCH=1 sbatch --export=ALL,TASK=TossBall,LEN=10,N=80,NENV=6 env/record_hpc.sbatch` |
| 录制并发布到看板 | `MODE=expert TASK=WipeWindow N=6 NENV=6 bash env/viewer_publish.sh` |
| π₀.₅ 评测 | 见 `eval_hpc.sbatch` 顶部注释（`CFG`、`CKPT`、`TASK`、`ROLLOUTS`） |
| 多任务训练 | `sbatch --export=ALL,CONFIG=<cfg>,REPO=<am_bench 下仓库名>,EXP=<实验名> env/train_mt.sbatch` |
| 单次 GPU 调试 | `sbatch --export=ALL,SCRIPT=env/xxx.py,ARGS='...' env/gpu_dbg.sbatch` |
| CPU 分析 | `sbatch --export=ALL,CMD='python ...' env/cpu_run.sbatch` |

## 回放看板

录制和评测的三视角回放通过 `viewer_publish.sh` 转换、打包并同步到看板服务器。看板地址和服务器登录方式向项目负责人索取。每个任务在看板上只保留最新的成功专家版本。
