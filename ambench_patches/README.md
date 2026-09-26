# AM-Bench 补丁与实验脚本

本目录是本项目对 AM-Bench 的全部改动，以及在 HPC 上做实验用到的脚本。AM-Bench 上游为 [ambench/ambench](https://github.com/ambench/ambench)，Apache-2.0 许可；改动基于 commit `60bf5b73041df4eab571f7d9f0a297aeecbf2e0d`。

## 应用补丁

```bash
cd <ambench 仓库>
git checkout 60bf5b73041df4eab571f7d9f0a297aeecbf2e0d
git apply <本仓库>/ambench_patches/ambench_fixes.diff
# 两个 USD 资产是二进制文件，没有包含在 diff 里，用脚本重新生成：
#   frame.usd：给把手短柱加碰撞体
python scripts/patch_frame_usd.py        # 需在 Isaac Sim 的 Python 环境中运行（pxr）
# container.usd 的修改最终没有生效，空心桶改在运行时实现（base_env.py 中的 _make_hollow_bin），无需处理
```

`ambench_fixes.diff` 覆盖的文件：

| 文件 | 改动 |
|---|---|
| `robots/fa_hexa.py` | 夹爪宽度换算函数；手指驱动刚度 100 → 1000 |
| `tasks/base/base_env.py` | 抓取点偏移参数；腕部六维力观测 `ee_wrench`；空心桶 `_make_hollow_bin` |
| `tasks/toss_ball/*` | 抓取点 0.13 m；球质量 0.05 kg；空心桶 |
| `tasks/frame_assembly/*` | 框架质量 0.3 kg；调试打印 |
| `tasks/lemon_harvesting/lemon_harvesting_env.py` | 空心桶 |
| `tasks/wipe_window/wipe_window_env.py` | 调试打印（接触力、腕部力、指尖距离），环境变量 `AMBENCH_WIPE_DEBUG=1` 打开 |
| `policies/scripted/toss_ball.py` | 解析弹道专家 |
| `policies/scripted/wipe_window.py` | 腕部力伺服专家 |
| `policies/scripted/frame_assembly.py` | 到墙后重规划（诊断阶段加入，根因修复后不再需要） |
| `scripts/data/record_demos_scripted.py` | 录制相关小改动 |
| `scripts/data/export_lerobot_to_openpi.py` | 识别 `ee_wrench`，`--include_wrench` 开关 |
| `ambench_learn/policies/pi/eval.py` | 专家在环回放 `--expert-policy`、视频录制等 |

每项改动的原因见 `docs/仿真修复清单.md`。

## scripts/

HPC 上 `$A/env/` 的精选副本。路径硬编码为 `/public/home/liaodl/ambench`，使用前改成自己的目录。

| 类别 | 脚本 |
|---|---|
| 环境 | `hpc_env.sh`（openpi 环境与代理）、`isaac_run.sh`、`isaac_py.sh`（Isaac Sim 容器） |
| 作业模板 | `record_hpc.sbatch`（录制）、`eval_hpc.sbatch`（π₀.₅ 评测）、`eval_hpc_opendm.sbatch`（DM0.5 评测）、`train_mt.sbatch`（多任务训练，含归一化统计）、`train_pi05.sbatch`、`st2cam_train.sbatch`（单任务双相机训练）、`gpu_dbg.sbatch`（单次 GPU 调试）、`cpu_run.sbatch`（CPU 分析） |
| 数据 | `st2cam_export.sh`、`export_new3.sh`、`export_clean12.sh`（导出为 openpi 格式并合并）、`merge_openpi_datasets.py`、`check_parquet.py`（检查写坏的文件）、`repair_session.py`（隔离写坏的集）、`launch_rerecord9.sh`（9 任务重录） |
| 环境修复 | `patch_grasp.py`、`patch_frame_usd.py`、`patch_hollow_bin.py`、`patch_wrench_obs.py`、`patch_wipe_servo*.py`、`patch_container_usd.py`（最终未采用） |
| 诊断与探针 | `toss_grasp_dbg.py`（逐步打印球、手指、机体几何，支持单轴运动对照）、`wrench_smoke.py`（腕部力与海绵传感器对比）、`frame_grip.py`、`toss_probe.py`、`audit_assets.py`、`probe_usd.py`（检查 USD 碰撞体与材质） |
| 看板 | `viewer_publish.sh`（录制或评测 → 转换 → 打包 → 同步），服务器地址需自行配置 |
| 换基座 | `convert_opendm.py`、`opendm_ambench_server.py`、`convert_beingh.py`、`beingh_ambench_server.py` |
| 清理 | `prune3.sh`（先不加参数查看清单，再 `DO=1` 执行） |

`patch_*.py` 是一次性脚本，按顺序作用在 HPC 上的源码，效果已经包含在 `ambench_fixes.diff` 里；保留它们是为了记录每次修改的意图。
