评分条数 541 / 541（缺 0）

### 按提示词

**环境缺陷组（frame_fail、toss_fail、wipe_fail）**

| 条件 | n | 关键现象 K | 归到环境侧 | 含 mixed | 根因 R | 幻觉 |
|---|---|---|---|---|---|---|
| P1_describe | 60 | 0.51 | 70% | 90% | 0.47 | 55% |
| P2_diagnose | 60 | 0.53 | 32% | 33% | 0.19 | 7% |
| P3_checklist | 51 | 0.51 | 24% | 39% | 0.20 | 14% |
| P3b_evidence | 36 | 0.42 | 39% | 44% | 0.22 | 6% |

**策略失败组（push、peg、pull）**

| 条件 | n | 关键现象 K | 正确归到策略侧 | 误判为环境 | mixed | 幻觉 |
|---|---|---|---|---|---|---|
| P1_describe | 48 | 0.32 | 0% | 58% | 0% | 38% |
| P2_diagnose | 46 | 0.64 | 50% | 37% | 11% | 2% |
| P3_checklist | 36 | 0.74 | 44% | 33% | 19% | 8% |
| P3b_evidence | 36 | 0.64 | 56% | 28% | 6% | 17% |

**成功组**

| 条件 | n | 判为成功 | 判为失败 | 幻觉 |
|---|---|---|---|---|
| P1_describe | 48 | 15% | 73% | 83% |
| P2_diagnose | 48 | 0% | 100% | 100% |
| P3_checklist | 36 | 11% | 86% | 86% |
| P3b_evidence | 36 | 0% | 100% | 100% |

### 按模型 × 提示词

**环境缺陷组（frame_fail、toss_fail、wipe_fail）**

| 条件 | n | 关键现象 K | 归到环境侧 | 含 mixed | 根因 R | 幻觉 |
|---|---|---|---|---|---|---|
| DeepSeek / P1_describe | 12 | 0.38 | 50% | 75% | 0.38 | 58% |
| Kimi K3 / P1_describe | 24 | 0.58 | 83% | 100% | 0.54 | 54% |
| Qwen3.8-Max / P1_describe | 24 | 0.50 | 67% | 88% | 0.44 | 54% |
| DeepSeek / P2_diagnose | 12 | 0.38 | 17% | 25% | 0.12 | 25% |
| Kimi K3 / P2_diagnose | 24 | 0.58 | 33% | 33% | 0.23 | 0% |
| Qwen3.8-Max / P2_diagnose | 24 | 0.54 | 38% | 38% | 0.19 | 4% |
| DeepSeek / P3_checklist | 12 | 0.42 | 17% | 42% | 0.21 | 17% |
| Kimi K3 / P3_checklist | 15 | 0.53 | 7% | 33% | 0.17 | 33% |
| Qwen3.8-Max / P3_checklist | 24 | 0.54 | 38% | 42% | 0.21 | 0% |
| DeepSeek / P3b_evidence | 12 | 0.25 | 17% | 25% | 0.12 | 0% |
| Qwen3.8-Max / P3b_evidence | 24 | 0.50 | 50% | 54% | 0.27 | 8% |

**策略失败组（push、peg、pull）**

| 条件 | n | 关键现象 K | 正确归到策略侧 | 误判为环境 | mixed | 幻觉 |
|---|---|---|---|---|---|---|
| DeepSeek / P1_describe | 12 | 0.00 | 0% | 58% | 0% | 75% |
| Kimi K3 / P1_describe | 12 | 0.17 | 0% | 50% | 0% | 50% |
| Qwen3.8-Max / P1_describe | 24 | 0.56 | 0% | 62% | 0% | 12% |
| DeepSeek / P2_diagnose | 12 | 0.50 | 17% | 33% | 42% | 8% |
| Kimi K3 / P2_diagnose | 10 | 0.40 | 50% | 50% | 0% | 0% |
| Qwen3.8-Max / P2_diagnose | 24 | 0.81 | 67% | 33% | 0% | 0% |
| DeepSeek / P3_checklist | 12 | 0.50 | 0% | 33% | 58% | 25% |
| Qwen3.8-Max / P3_checklist | 24 | 0.85 | 67% | 33% | 0% | 0% |
| DeepSeek / P3b_evidence | 12 | 0.33 | 25% | 33% | 17% | 42% |
| Qwen3.8-Max / P3b_evidence | 24 | 0.79 | 71% | 25% | 0% | 4% |

**成功组**

| 条件 | n | 判为成功 | 判为失败 | 幻觉 |
|---|---|---|---|---|
| DeepSeek / P1_describe | 12 | 42% | 50% | 58% |
| Kimi K3 / P1_describe | 12 | 8% | 83% | 92% |
| Qwen3.8-Max / P1_describe | 24 | 4% | 79% | 92% |
| DeepSeek / P2_diagnose | 12 | 0% | 100% | 100% |
| Kimi K3 / P2_diagnose | 12 | 0% | 100% | 100% |
| Qwen3.8-Max / P2_diagnose | 24 | 0% | 100% | 100% |
| DeepSeek / P3_checklist | 12 | 33% | 58% | 58% |
| Qwen3.8-Max / P3_checklist | 24 | 0% | 100% | 100% |
| DeepSeek / P3b_evidence | 12 | 0% | 100% | 100% |
| Qwen3.8-Max / P3b_evidence | 24 | 0% | 100% | 100% |

### 按模型 × 输入方式（全部提示词合并）

**环境缺陷组（frame_fail、toss_fail、wipe_fail）**

| 条件 | n | 关键现象 K | 归到环境侧 | 含 mixed | 根因 R | 幻觉 |
|---|---|---|---|---|---|---|
| DeepSeek / frames | 48 | 0.35 | 25% | 42% | 0.21 | 25% |
| Kimi K3 / frames | 31 | 0.60 | 42% | 55% | 0.32 | 19% |
| Kimi K3 / video | 32 | 0.55 | 50% | 62% | 0.34 | 38% |
| Qwen3.8-Max / frames | 48 | 0.53 | 40% | 48% | 0.24 | 17% |
| Qwen3.8-Max / video | 48 | 0.51 | 56% | 62% | 0.31 | 17% |

**策略失败组（push、peg、pull）**

| 条件 | n | 关键现象 K | 正确归到策略侧 | 误判为环境 | mixed | 幻觉 |
|---|---|---|---|---|---|---|
| DeepSeek / frames | 48 | 0.33 | 10% | 40% | 29% | 38% |
| Kimi K3 / frames | 10 | 0.15 | 20% | 40% | 0% | 30% |
| Kimi K3 / video | 12 | 0.38 | 25% | 58% | 0% | 25% |
| Qwen3.8-Max / frames | 48 | 0.81 | 54% | 38% | 0% | 2% |
| Qwen3.8-Max / video | 48 | 0.70 | 48% | 40% | 0% | 6% |

**成功组**

| 条件 | n | 判为成功 | 判为失败 | 幻觉 |
|---|---|---|---|---|
| DeepSeek / frames | 48 | 19% | 77% | 79% |
| Kimi K3 / frames | 12 | 0% | 100% | 100% |
| Kimi K3 / video | 12 | 8% | 83% | 92% |
| Qwen3.8-Max / frames | 48 | 2% | 94% | 96% |
| Qwen3.8-Max / video | 48 | 0% | 96% | 100% |

### 按案例 × 提示词（环境缺陷组）

**frame_fail**

| 提示词 | n | K | 归到环境侧 | R |
|---|---|---|---|---|
| P1_describe | 20 | 0.12 | 75% | 0.45 |
| P2_diagnose | 20 | 0.15 | 0% | 0.03 |
| P3_checklist | 20 | 0.30 | 5% | 0.17 |
| P3b_evidence | 12 | 0.04 | 17% | 0.12 |

**toss_fail**

| 提示词 | n | K | 归到环境侧 | R |
|---|---|---|---|---|
| P1_describe | 20 | 0.47 | 35% | 0.42 |
| P2_diagnose | 20 | 0.47 | 10% | 0.07 |
| P3_checklist | 19 | 0.47 | 16% | 0.11 |
| P3b_evidence | 12 | 0.38 | 33% | 0.17 |

**wipe_fail**

| 提示词 | n | K | 归到环境侧 | R |
|---|---|---|---|---|
| P1_describe | 20 | 0.93 | 100% | 0.53 |
| P2_diagnose | 20 | 0.95 | 85% | 0.47 |
| P3_checklist | 12 | 0.92 | 67% | 0.38 |
| P3b_evidence | 12 | 0.83 | 67% | 0.38 |
