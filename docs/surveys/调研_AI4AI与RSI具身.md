# AI4AI / 递归自我改进（RSI）及其在具身智能（机器人 VLA）中的应用——文献调研

调研范围：用 AI（VLM/LLM Agent）自动化"观察机器人模型失败 → 修改/新建仿真任务、资产、脚本专家 → 录制演示数据 → 训练 VLA → 评测 → 再观察"这一闭环的相关文献。覆盖通用 AI4AI/RSI（A）、具身任务与场景生成（B）、自动奖励/专家生成（C）、自动数据生成与增强（D）、模型自我改进（E）、自动评测与失败诊断（F）、空中操作仿真基准（G），共 7 个方向。

> 说明：本报告由多路并行文献检索汇总而成，所有链接均经过搜索核实；无法确认真实存在的论文/编号均标注"待核实"，不做编造。

---

## A. 通用 AI4AI / 递归自我改进（RSI）

### 1. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery
- 作者/机构：Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, David Ha（Sakana AI / FLAIR Oxford / UBC）
- 年份：2024（v1，2024-08）；v2 于 2025 年发布，三篇论文投稿 ICLR Workshop，其中一篇被接收
- 链接：[arXiv:2408.06292](https://arxiv.org/abs/2408.06292)；[代码](https://github.com/SakanaAI/AI-Scientist)；[官方博客](https://sakana.ai/ai-scientist/)
- 核心方法：LLM 自主完成"提出想法→写代码→跑实验→画图→写论文→模拟同行评审"全流程，闭环内含自我修正（根据评审反馈迭代论文/实验）。
- 自动化 vs 人工：想法生成、代码实现、实验执行、结果可视化、论文撰写、评审打分全部自动化。仍靠人：研究方向的模板/初始代码库由人提供；最终论文质量与真实科学价值仍需人工把关；安全沙箱、防止实验运行失控靠人设计。
- 关键结果数字：单篇论文成本约 $15；自动评审判定部分论文超过顶会接收阈值；v2 有 1 篇被真实 ICLR workshop 接收。
- 与具身闭环的差距：这是"科研任务"上的端到端自动化范式（idea→实验→论文），但实验环境是通用代码/ML 基准，不涉及物理仿真资产、机器人专家脚本或真实失败模式诊断，无法直接迁移到"改任务/资产/专家"这一步。

### 2. AlphaEvolve: A coding agent for scientific and algorithmic discovery
- 作者/机构：Google DeepMind（Novikov et al.）
- 年份：2025（技术报告，2025-05）
- 链接：[arXiv:2506.13131](https://arxiv.org/abs/2506.13131)；[博客](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- 核心方法：用 Gemini Flash（广度）+Gemini Pro（深度）做进化式代码生成，配合自动化评估器（可执行、可验证的打分函数）对候选程序做选择/变异，持续进化出更优算法。
- 自动化 vs 人工：候选解生成、变异/重组、评估打分、种群进化全自动，甚至用于加速训练 Gemini 自身（自举）。仍靠人：每个任务的"自动评估器"（fitness function）需要人工设计并保证正确性；任务本身（如调度算法、矩阵乘法电路简化）由人指定。
- 关键结果数字：发现数据中心调度算法效率提升；硬件加速器电路简化；帮助加速训练支撑自身的 LLM。
- 与具身闭环的差距：AlphaEvolve 证明了"自动评估器+LLM 进化"范式在有明确可验证目标函数的领域可行，但机器人任务成功判定往往没有现成的可执行 fitness function（需要 VLM 做视觉判断），这是直接套用到具身闭环的主要障碍。

### 3. Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents
- 作者/机构：Jenny Zhang, Shengran Hu（Jeff Clune 组，UBC）+ Cong Lu, Robert Lange（Sakana AI）
- 年份：2025-05
- 链接：[arXiv:2505.22954](https://arxiv.org/abs/2505.22954)；[代码](https://github.com/jennyzzt/dgm)
- 核心方法：Agent 自己重写自己的代码（agent 架构/prompt/工具使用逻辑），每次改动都在编码基准（SWE-bench、Polyglot）上实证验证，用类似开放式进化的档案（archive）保留多条谱系而非单一贪心路径。
- 自动化 vs 人工：自我代码修改的生成、变异、在基准上的验证、谱系选择全自动。仍靠人：基准任务集（SWE-bench/Polyglot）本身由人构建；安全护栏（防止无限制自我修改失控）需人工设计；初始 agent 骨架由人给出。
- 关键结果数字：SWE-bench 成功率从 20.0% 提升到 50.0%；Polyglot 成功率从 14.2% 提升到 30.7%。
- 与具身闭环的差距：DGM 的"失败观察→修改自身→验证"循环结构与目标闭环高度类似，但作用对象是"agent 自己的代码"而不是"机器人训练所需的仿真任务/资产/专家脚本"；其验证信号（能否通过编码基准）是现成的，而具身场景的验证信号（仿真成功率）需要额外构建。这是结构上最接近但领域完全不同的参照系。

### 4. STaR: Bootstrapping Reasoning With Reasoning（Self-Taught Reasoner）
- 作者/机构：Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah Goodman（Stanford）
- 年份：2022-03
- 链接：[arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- 核心方法：模型自己生成推理链（rationale），用最终答案是否正确来筛选/回填训练数据，再微调自己，迭代提升推理能力（自举式自训练）。
- 自动化 vs 人工：数据生成（rationale 采样）、正确性过滤、再训练全自动，形成"生成数据→自我训练"的最小闭环。仍靠人：初始少量带 rationale 的示例仍由人提供；奖励信号（答案是否正确）依赖有标准答案的题目集，本身是人构造的数据集。
- 关键结果数字：在 CommonsenseQA 等任务上显著提升泛化性能（论文以准确率提升曲线呈现，无单一汇总数字）。
- 与具身闭环的差距：STaR 展示了"用自己生成的数据+结果过滤来自我训练"这一最基本模式，是后续所有 self-improving 工作的雏形，但没有"环境/任务生成"这一环——它假设任务集固定，只在数据/训练层面自举，不涉及为了暴露模型弱点而主动构造新任务。

### 5. Automated Design of Agentic Systems（ADAS）/ Meta Agent Search
- 作者/机构：Shengran Hu, Cong Lu, Jeff Clune（UBC / Sakana AI，部分作者与 DGM 重合）
- 年份：2024-08
- 链接：[arXiv:2408.08435](https://arxiv.org/abs/2408.08435)
- 核心方法：提出"元 agent"用代码的形式迭代设计新的 agentic 系统（新的 prompt、工具调用、控制流组合），并维护一个不断增长的历史发现档案（archive）作为上下文，让搜索越来越聪明。
- 自动化 vs 人工：agent 架构的生成、评估、档案维护、跨任务/跨模型迁移验证全自动。仍靠人：目标任务集（coding、science、math benchmark）由人给定；评估指标（在这些 benchmark 上的准确率）由人预先定义。
- 关键结果数字：搜索出的新 agent 设计在多个 domain 上显著超越人工设计的 SOTA agent，且能跨领域、跨模型迁移仍保持性能优势（多基准对比，无单一汇总百分比）。
- 与具身闭环的差距：ADAS 是 DGM 的前身/同源工作，同样是"自动改进 agent 设计"而非"改进机器人训练所需的仿真环境/数据"；可类比的迁移思路是把"agent 架构搜索"换成"仿真任务/专家脚本搜索"，但目前没有工作做这个替换。

### 6. Reinforced Self-Training（ReST）for Language Modeling
- 作者/机构：Caglar Gulcehre 等（Google DeepMind）
- 年份：2023-08
- 链接：[arXiv:2308.08998](https://arxiv.org/abs/2308.08998)
- 核心方法：Grow（用当前策略采样生成数据集）+Improve（用离线 RL 在该数据集上多轮训练）交替进行，数据生产与策略优化解耦，可复用数据、降低在线 RL 成本。
- 自动化 vs 人工：样本生成、离线训练全自动，形成生成-训练交替的闭环。仍靠人：奖励模型/评分函数（用于筛选生成样本质量）需要人工训练或指定；最初策略与任务分布由人给定。
- 关键结果数字：在机器翻译任务上用自动指标和人工评估都显示翻译质量显著提升（论文聚焦相对提升，无单一权威数字）。
- 与具身闭环的差距：ReST 的 Grow/Improve 循环结构（自己产数据→自己训练→再产数据）与"rollout→训练→再 rollout"部分吻合，但同样缺少"环境/任务生成"环节，且应用在文本域，奖励信号明确（翻译质量指标），而机器人失败诊断更复杂。

### 7. RSI 综述与理论/安全讨论
- **综述**：一篇系统性调研 1,250 篇 2024–2026 arXiv 论文的综述性文章，沿两个轴分类——"系统改进什么"（输出/部署时适应/训练数据/AI 自身研究）和"闭环程度"（人在环 vs. 完全闭环），区分"有界自我精炼"与"开放式 RSI"。链接：[arXiv:2607.07663](https://arxiv.org/abs/2607.07663)（作者信息与准确提交时间**待核实**）。核心结论：绝大多数被调研工作仍是"有界自我精炼"，真正闭环修改自己评估器的开放式 RSI 极少；"治理级别的自我改进度量"被认为是该领域最欠缺的空白。这一分类框架可直接借用来给本项目自己的具身闭环打分，其核心结论与本报告在具身领域观察到的空白高度一致。
- **安全/理论讨论（补充，细节待核实）**：Recursive Criticality of AI Self-Improvement（[arXiv:2609.00137](https://arxiv.org/abs/2609.00137)）；Self-reference in LLMs: introspection threshold for recursive self-improvement（[arXiv:2607.04277](https://arxiv.org/html/2607.04277v1)）。这两篇标注为"待核实细节"，仅作为 RSI 安全讨论存在性的旁证。

**说明**：检索未找到高度贴合且权威的单篇"AutoML 自动化数据引擎"代表作（该领域论文众多但分散，不单独展开）；Self-play 微调代表作（如 SPIN）本次未充分核实，暂不列入，避免编造。

---

## B. 具身：自动任务/场景生成

| # | 论文 | 作者/机构 | 年份 | 链接 |
|---|------|----------|------|------|
| 1 | RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation | Yufei Wang 等，CMU | 2023（ICML 2024） | [arXiv:2311.01455](https://arxiv.org/abs/2311.01455) |
| 2 | GenSim: Generating Robotic Simulation Tasks via Large Language Models | Lirui Wang 等，MIT/CMU | 2023 | [arXiv:2310.01361](https://arxiv.org/abs/2310.01361) |
| 3 | Gen2Sim: Scaling up Robot Learning in Simulation with Generative Models | Pushkal Katara, Zhou Xian, Katerina Fragkiadaki，CMU | 2023（ICRA 2024） | [arXiv:2310.18308](https://arxiv.org/abs/2310.18308) |
| 4 | GenSim2: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs | Pu Hua, Minghuan Liu 等 | 2024 | [arXiv:2410.03645](https://arxiv.org/abs/2410.03645) |
| 5 | BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation | Stanford（Li Fei-Fei 组）等 | 2024 | [arXiv:2403.09227](https://arxiv.org/abs/2403.09227) |
| 6 | Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition | 相关团队 | 2023 | [arXiv:2307.14535](https://arxiv.org/abs/2307.14535) |
| 7 | ProcTHOR: Large-Scale Embodied AI Using Procedural Generation | Matt Deitke 等，AI2/UW | 2022（NeurIPS 2022） | [arXiv:2206.06994](https://arxiv.org/abs/2206.06994) |
| 8 | Holodeck: Language Guided Generation of 3D Embodied AI Environments | Yue Yang 等，AI2/UPenn | 2023（CVPR 2024） | [arXiv:2312.09067](https://arxiv.org/abs/2312.09067) |
| 9 | EmbodiedGen / EmbodiedGen V2 | Horizon Robotics 团队 | 2025 | [arXiv:2506.10600](https://arxiv.org/abs/2506.10600)；[arXiv:2607.07459](https://arxiv.org/abs/2607.07459) |
| 10 | RoboVerse: Towards a Unified Platform, Dataset and Benchmark for Scalable and Generalizable Robot Learning | 多机构联合 | 2025 | [arXiv:2504.18904](https://arxiv.org/abs/2504.18904) |
| 11 | Infinigen / Infinigen Indoors | Princeton Vision & Learning Lab | 2023/2024 | [arXiv:2306.09310](https://arxiv.org/abs/2306.09310)；[arXiv:2406.11824](https://arxiv.org/abs/2406.11824) |
| 12 | BBSEA: An Exploration of Brain-Body Synchronization for Embodied Agents | Sizhe Yang 等 | 2024 | [arXiv:2402.08212](https://arxiv.org/abs/2402.08212) |
| 13 | Genesis: A Generative and Universal Physics Engine for Robotics and Beyond | Zhou Xian 等，20+ 实验室联合项目 | 2024 年 12 月发布 | [GitHub](https://github.com/Genesis-Embodied-AI/genesis-world)（**待核实**：未找到正式配套 arXiv 论文） |
| 14 | SAGE: Scalable Agentic 3D Scene Generation for Embodied AI | 作者机构**待核实** | 2026 | [arXiv:2602.10116](https://arxiv.org/abs/2602.10116) |

**逐条要点：**

1. **RoboGen** — "propose–generate–learn"自引导闭环：基础/生成模型自动提出任务、生成场景、生成监督信号（RL 奖励或运动规划）。*自动化*：任务提议、场景组装、监督信号生成。*仍靠人*：底层 3D 资产库（来自 Objaverse/PartNet-Mobility 等既有数据集）、机器人本体预设。与目标闭环的差距：是**开环批量生成**，没有"策略失败反推任务"的机制。

2. **GenSim** — LLM（GPT-4）以目标导向/探索式两种模式生成任务代码，把已有基准扩充 10 倍以上（100+ 任务）。*自动化*：任务代码与课程生成。*仍靠人*：初始种子任务库人工编写，任务合理性主要靠 LLM 自检+仿真跑通而非策略失败驱动。差距：驱动力是"覆盖多样性"而非"策略在具体任务上失败"。

3. **Gen2Sim** — 2D 图像经扩散模型提升为 3D 资产，LLM 推断物理参数，chain-of-thought 把资产映射到任务描述/时间分解/Python 奖励函数（兼具 B、C 部分性质）。*自动化*：资产生成、任务/奖励函数生成。*仍靠人*：人工开发的资产仍与生成资产混用。差距：开环"资产→任务→奖励"流水线，无失败驱动的定向生成。

4. **GenSim2** — 多模态/推理 LLM 生成含关节物体的长时程任务，配合规划器/RL 生成演示，训练策略并做 sim-to-real 迁移。规模：最多 100 个关节任务、200 物体。*仍靠人*：资产库、任务模板类别预先框定。差距：广度优先批量扩充，没有针对特定 VLA 失败案例定向补数据，也没有真实失败信息回传修改仿真任务。

5. **BEHAVIOR-1K / "Scaling Up and Distilling Down"** — 前者是 1000 个日常活动、50 场景、9000+ 物体的人工设计+半自动基准（非自动任务生成引擎）；后者用 LLM 做高层规划、采样式规划器自动生成轨迹、蒸馏为语言条件视觉运动策略，五个领域绝对成功率平均提升 33.2%。*仍靠人*：1000 活动定义、场景/物体标注基本人工完成，目标任务集合固定不变，无策略失败反馈来新增/修改任务。

6. **ProcTHOR** — 程序化生成任意规模可交互房屋（实验用 10000 个），全自动布局/物体摆放采样。*仍靠人*：房间模板、AI2-THOR 资产类别体系需人工设计。结果：10000 房屋训练模型在 6 个基准（Habitat、AI2-THOR Rearrangement、RoboTHOR 等）SOTA。差距：纯场景多样性扩充引擎，不涉及任务专家/演示生成，不针对已部署模型失败模式。

7. **Holodeck** — GPT-4 提供场景常识，检索 Objaverse 资产，GPT-4 生成空间关系约束并优化布局，全自动依文本 prompt 生成 3D 环境。*仍靠人*：Objaverse 资产库人工/众包收集，质量由人类评估者打分（而非策略性能反馈）。差距：面向导航/场景多样性，不涉及操作任务专家或失败驱动迭代。

8. **EmbodiedGen / V2** — 六模块生成式 3D 世界引擎（图生 3D、文生 3D、纹理、关节物体、场景、布局生成），产出 URDF 格式资产；V2 做到 agentic、跨仿真器统一表示、任务驱动世界生成。*自动化*：资产/场景/任务驱动世界生成均由生成式 AI+agentic 流程完成，是本次检索中"任务驱动"描述最接近目标闭环的候选底座之一。*仍靠人*：生成什么样任务/场景的"意图"仍由人类下达，未见"策略失败自动触发生成请求"机制。差距：是重要的基础设施型工作而非完整闭环——触发生成的输入是人类意图而非模型失败信号。

9. **RoboVerse** — MetaSim 统一多仿真器/多本体抽象层+合成数据集+统一基准，服务模仿学习/RL/世界模型/sim2real 系统性评测。是"平台标准化"工作，不直接解决"自动修改任务"，可作为评测环节基础设施。

10. **Infinigen / Indoors** — 纯规则驱动程序化生成（无学习/LLM），自然场景与室内场景零外部数据源无限变体。*仍靠人*：生成语法/规则本身人工编写。差距：图形学路线，与 LLM 驱动任务发现是不同技术路线，不涉及策略失败反馈。

11. **BBSEA** — LLM"大脑"基于场景图提出与场景/物理限制兼容的任务，机器人"身体"试错学习并把反馈回传给"大脑"，形成持续技能习得循环。**这是本组文献里机制上最接近"失败观察驱动任务生成"的一个**：执行反馈确实回传影响后续任务提议。但差距明显：(a) 小规模单场景验证，未涉及修改/新建仿真资产这一步；(b) 无大规模 VLA 训练-评测完整闭环；(c) 反馈粒度粗（"提出下一个可行任务"层面），无精细的失败诊断-归因-定向修任务/专家流程。

12. **Genesis** — 物理引擎+光真实渲染+"生成数据引擎"（自然语言 prompt 转多模态数据），2024 年 12 月发布，速度号称远超 Isaac 系列。**未找到正式同名 arXiv 研究论文**，仅有项目主页/GitHub/文档，标注**待核实**。

13. **SAGE（2026）** — agentic 框架：多个 layout/物体生成器+critic 模块联合工作，critic 评估语义合理性/视觉真实感/物理稳定性，迭代"自我修正"场景直至满足意图与物理有效性。*自动化*：生成-评估-修正迭代闭环全自动。*仍靠人*：最初任务意图由人类给出，critic 标准是通用物理/语义合理性，**不是 policy 在该场景中是否失败**。差距：方法论上是"生成-自我批评-再生成"agentic pattern，离目标更近一步，但驱动力不是下游 VLA 表现。作者/机构/数字待核实。

---

## C. 具身：自动奖励/专家/技能生成

本节聚焦"用 LLM/VLM 自动写奖励函数、脚本策略（scripted policy/expert）或规划代码"这一类工作。核心问题：**"写专家"这个环节被自动化到什么程度**，以及这类自动化是否已经和"下游 VLA 训练失败"形成闭环。

### 1. Eureka: Human-Level Reward Design via Coding Large Language Models
- 作者/机构：Yecheng Jason Ma 等，NVIDIA / UPenn / Caltech / UT Austin
- 年份：2023（ICLR 2024）
- 链接：[arXiv:2310.12931](https://arxiv.org/abs/2310.12931)；[项目页](https://eureka-research.github.io/)；[代码](https://github.com/eureka-research/Eureka)
- 核心方法：把仿真环境源码+任务的自然语言描述直接喂给 GPT-4，零样本生成可执行的奖励函数代码；随后在 GPU 并行仿真中评估该奖励训练出的策略，把训练统计量反馈给 LLM 做"奖励反思"，反复进化奖励代码。
- 自动化 vs 人工：**奖励函数的编写和迭代改进完全自动化**（LLM 写代码、读训练曲线、重写）。仍靠人：①环境代码和任务描述由人预写；②仿真场景/机器人本体是已有的、人搭建的 Isaac Gym 环境；③下游是 RL（PPO）不是 VLA，评测指标固定，没有真正的"VLA 失败观察"反馈环。
- 关键结果：29 个开源 RL 环境（10 种机器人形态）中 83% 的任务上优于人工设计奖励，平均归一化提升 52%。
- 与我们闭环的差距：闭环止步于奖励设计本身，不涉及自动生成新任务/新资产，不服务于 VLA 模仿学习数据采集，更没有拿"下游 VLA 失败模式"作为触发信号。

### 2. DrEureka: Language Model Guided Sim-to-Real Transfer
- 作者/机构：Yecheng Jason Ma 等，NVIDIA / UPenn（Eureka 团队后续工作）
- 年份：2024（RSS 2024）
- 链接：[arXiv:2406.01967](https://arxiv.org/abs/2406.01967)；[项目页](https://eureka-research.github.io/dr-eureka/)
- 核心方法：用 LLM 同时自动生成奖励函数、安全约束，以及基于策略训练中物理量表现反推出的**域随机化参数分布**，把 sim-to-real 中两个最耗人力的环节都交给 LLM。
- 自动化 vs 人工：奖励与 DR 参数生成/调整自动化；但任务本身（四足行走、瑜伽球平衡等）仍由人指定，物理引擎和机器人模型给定，真实机器人最终成功率验证仍靠人。
- 关键结果：四足运动和灵巧操作任务上，无需人工迭代设计即可发现与人工配置相当或更优的 sim-to-real 配置，能独立解决人类未手工调过的新任务。
- 与我们闭环的差距：是本节中离"观察失败→自动重写专家"最近的工作之一，但目标仍是 RL 用奖励/DR，而非 VLA 演示脚本，也没有自动修改任务/资产的能力。

### 3. Text2Reward: Reward Shaping with Language Models for Reinforcement Learning
- 作者/机构：Tianbao Xie 等（清华、UIUC 等）
- 年份：2023（ICLR 2024）
- 链接：[arXiv:2309.11489](https://arxiv.org/abs/2309.11489)
- 核心方法：给定自然语言目标和环境的紧凑符号表示，LLM 生成可解释的稠密奖励代码；支持人类语言反馈迭代重写。
- 自动化 vs 人工：奖励代码生成自动化；迭代改进环节**既可全自动（训练成功率回灌 LLM），也允许人类语言反馈介入**（论文强调是可选的人机协同）。环境/任务列表由人预定义。
- 关键结果：17 个操作任务中 13 个上，生成奖励训练出的策略成功率/收敛速度与专家手写奖励相当或更好；运动任务上学出 6 种新颖行为，成功率超 94%。
- 与我们闭环的差距：止步于奖励代码层，不触及任务/资产生成，也不为 VLA 数据采集服务；"人类语言反馈"选项说明当前这类工作仍默认把"观察失败→给出修改意见"留给人。

### 4. Language to Rewards for Robotic Skill Synthesis（L2R）
- 作者/机构：Wenhao Yu 等，Google DeepMind
- 年份：2023（CoRL 2023）
- 链接：[arXiv:2306.08647](https://arxiv.org/abs/2306.08647)；[博客](https://research.google/blog/language-to-rewards-for-robotic-skill-synthesis/)
- 核心方法：LLM 把自然语言指令翻译成奖励参数代码，MuJoCo MPC 在线求解满足奖励的轨迹（不训练权重）；支持用户实时语言反馈纠正机器人行为。
- 自动化 vs 人工：奖励生成+轨迹优化自动化；**任务指令与迭代反馈由人在回合中实时给出**——这是论文明确的人在环教学设计目标，而非无人闭环。
- 关键结果：完成四足"太空步"、机械臂多步放置任务等定性展示，未见大规模数值基线对比表。
- 与我们闭环的差距：是"人类在环教学"范式的代表，与我们目标（去掉人在环）方向相反，可作对照，说明多数此类工作默认人类实时参与纠正。

### 5. Code as Policies: Language Model Programs for Embodied Control
- 作者/机构：Jacky Liang, Wenlong Huang 等，Robotics at Google
- 年份：2022（ICRA 2023）
- 链接：[arXiv:2209.07753](https://arxiv.org/abs/2209.07753)；[项目页](https://code-as-policies.github.io/)
- 核心方法：LLM 作为代码补全器，给少量指令-策略代码 few-shot 示例，直接生成调用感知/控制 API 的 Python 程序作为机器人策略——本质是自动生成脚本专家。
- 自动化 vs 人工：策略代码生成自动化，具备组合泛化能力；但**底层控制原语 API 由人预先实现**，LLM 只编排调用；无失败后自动重写闭环，策略对错由人观察视频判断。
- 关键结果：多个真实机器人平台完成拾取放置、绘图、桌面整理等任务，展示零样本组合泛化（偏定性展示，未报告大规模基准成功率）。
- 与我们闭环的差距：最早证明"LLM 写脚本专家"可行的代表作之一，但停留在真实机器人零样本执行层面，未涉及仿真任务/资产生成，未用于批量采集 VLA 训练数据。

### 6. VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models
- 作者/机构：Wenlong Huang, Chen Wang 等（Stanford）
- 年份：2023（CoRL 2023）
- 链接：[arXiv:2307.05973](https://arxiv.org/abs/2307.05973)；[项目页](https://voxposer.github.io/)
- 核心方法：LLM 推理任务所需的 affordance 和约束并写代码调用 VLM 在 3D 体素空间合成"价值地图"，再用现成运动规划器在该价值地图上求解轨迹——LLM+VLM 联合生成隐式脚本专家，零样本、无需机器人交互数据或训练。
- 自动化 vs 人工：任务到轨迹全过程自动化；但需人工提供任务语言描述，运动规划器/控制接口给定，对 VLM 检测词表外的物体/关系仍会失败，需人工调整 prompt。
- 关键结果：仿真和真实机器人上完成大量日常操作任务，相比 CLIPort、Code as Policies 等基线有更好零样本泛化（具体数字**待核实**，建议查阅原文附录表格）。
- 与我们闭环的差距：目标是直接控制机器人完成任务，而非生成大规模演示数据训练独立 VLA，且没有观察 VLA 失败、返回修改任务/专家的机制。

**C 部分小结**：以上工作共同证明"给定任务描述，用 LLM/VLM 自动生成奖励函数或脚本专家代码"已经是相对成熟、被反复验证的技术（Eureka/DrEureka/Text2Reward 有"看训练结果重写"的自动迭代圈，最接近"自我修正"）。但没有一篇工作把这个重写循环的**触发信号**设计成"下游 VLA 模型在某个仿真基准上的失败案例"，也没有工作在专家/奖励重写之外同时自动新建/修改仿真任务和资产。C 类工作解决的是闭环五环节里"专家生成"这一环的自动化，但（1）没有和"任务/资产自动修改"打通，（2）触发信号是 RL 训练曲线或人类语言反馈，而非 VLA 模仿学习的评测失败，（3）几乎不产生"用于训练 VLA 的规模化演示数据集"这一最终产物——这是与目标闭环最主要的缺口。RoboGen（B 部分）把"LLM 生成任务+LLM/VLM 生成奖励或调用运动规划器生成演示"整合在一起，是 B/C 两部分的交叉点。

---

## D. 具身：自动数据生成与增强

| 论文 | 作者/机构 | 年份 | 链接 |
|---|---|---|---|
| MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations | Mandlekar, Nasiriany 等（NVIDIA） | 2023（CoRL） | [arXiv:2310.17596](https://arxiv.org/abs/2310.17596) |
| DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning | Jiang, Xie 等（NVIDIA/UT Austin） | 2024→ICRA 2025 | [arXiv:2410.24185](https://arxiv.org/abs/2410.24185) |
| SkillMimicGen（SkillGen）: Automated Demonstration Generation for Efficient Skill Learning and Deployment | Garrett 等（NVIDIA） | 2024（CoRL） | [arXiv:2410.18907](https://arxiv.org/abs/2410.18907) |
| RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots | Nasiriany, Zhu 等（UT Austin/NVIDIA） | 2024（RSS） | [arXiv:2406.02523](https://arxiv.org/abs/2406.02523) |
| AutoRT: Embodied Foundation Models for Large Scale Orchestration of Robotic Agents | Ahn 等（Google DeepMind） | 2024 | [arXiv:2401.12963](https://arxiv.org/abs/2401.12963) |
| Robotic Skill Acquisition via Instruction Augmentation with Vision-Language Models（DIAL） | Xiao 等（Google） | 2022/2023 | [arXiv:2211.11736](https://arxiv.org/abs/2211.11736) |
| DexFlyWheel: A Scalable and Self-improving Data Generation Framework for Dexterous Manipulation | 中国团队（NeurIPS 2025） | 2025 | [arXiv:2509.23829](https://arxiv.org/abs/2509.23829) |

**核心方法与自动化/人工划分：**

- **MimicGen**：从 1–10 条人类演示中提取"以物体为中心"的子轨迹片段，通过坐标变换自动重放/拼接到新的物体位姿，批量生成上千条新演示。*自动化*：轨迹变换、碰撞检测重放、成功过滤。*人工*：仍需人工提供种子演示、人工定义任务/物体资产、人工设定物体位姿分布范围。不涉及任务本身的变更，只做同一任务内的轨迹增强。
- **DexMimicGen**：把 MimicGen 扩展到双臂+灵巧手，处理手指级接触约束。自动化程度与 MimicGen 相同；仍需为每个新任务/夹爪重新调参运动规划器。
- **SkillMimicGen**：把演示切分成可复用的"技能片段"（抓取、插入等），自动适配新场景并用自由空间运动拼接，24 个人类演示→2.4 万条演示，24 个任务变体，策略成功率平均提升 24%。*自动化*：技能分割、跨场景迁移、拼接；*人工*：技能边界的初始标注、任务集合仍是人工设计好的。
- **RoboCasa**：用生成式 AI（文生图/文生 3D+LLM）自动生成 120 个厨房场景、2500+物体资产，并用 LLM 辅助生成 100 个任务描述，再用 MimicGen 扩增出 10 万+轨迹。*自动化*：场景/资产的生成式合成、任务文本的 LLM 生成、轨迹级数据增强。*人工*：任务的原始种子演示仍由人类遥操作；任务的"是否成功"的奖励/校验逻辑是人工写的启发式代码，而非模型自动诊断失败后再生成新任务。
- **AutoRT**：用 VLM 感知环境+LLM 提出任务提议（"Robot Constitution"过滤安全性），编排 20+台真实机器人自主与教师遥操作混合采集，累计 7.7 万真实回合。*自动化*：任务提议、多机器人调度、安全过滤；*人工*：仍需人工远程干预/遥操作纠正、人工设计基础技能库（RT-1/RT-2 已具备的原子技能），失败后不会自动修改机器人技能模块或环境本身。
- **DIAL**：用 CLIP+LLM 把无标注的历史轨迹自动打上语言标签（LLM 生成大量语言复述，CLIP 做轨迹-语言匹配），从而免除昂贵人工标注——8 万条演示中 96.5% 的无标签数据被自动打标。*自动化*：标签生成与匹配；*人工*：种子标注集合、轨迹本身的采集仍是人工遥操作产生。
- **DexFlyWheel**（最贴近"飞轮"概念）：IL+残差 RL 训练策略→用策略在仿真中做 rollout 采集新轨迹→自动增强空间配置→喂回下一轮训练，迭代闭环。四个任务生成 2000+演示，仿真成功率 81.9%，真实世界（数字孪生迁移）双臂搬运成功率 78.3%。*自动化*：rollout 采集、数据增强、迭代训练全流程自动化，是这组工作里"自我改进闭环"最完整的一个。*人工*：任务集合固定为 4 个预先设计好的任务，物体资产、场景本身、奖励函数仍是人工设计，不会根据失败模式反过来创建新任务或修改资产库。

**与目标闭环的差距（D 部分整体）**：这一组工作全部聚焦"给定任务和场景，把演示数据做大做多"，即"数据扩增引擎"，而不是"任务/环境本身的迭代"。没有一篇论文出现"根据 VLA 失败模式反推应该造哪个新任务/新资产/新专家脚本"这一环——任务和场景空间在开始时就已经固定，自动化只发生在给定任务内部的轨迹/标签维度。DexFlyWheel 最接近"数据飞轮"，但飞轮转的是同一固定任务集合内的策略-数据迭代，并非任务本身的生成式迭代。

---

## E. 具身：模型自我改进

| 论文 | 作者/机构 | 年份 | 链接 |
|---|---|---|---|
| RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation | Bousmalis 等（Google DeepMind） | 2023 | [arXiv:2306.11706](https://arxiv.org/abs/2306.11706)；[DeepMind 博客](https://deepmind.google/blog/robocat-a-self-improving-robotic-agent/) |
| Self-Improving Embodied Foundation Models | Google DeepMind | 2025 | [arXiv:2509.15155](https://arxiv.org/pdf/2509.15155) |
| π*0.6: a VLA That Learns From Experience（含 RECAP 方法） | Physical Intelligence | 2025 | [arXiv:2511.14759](https://arxiv.org/html/2511.14759)；[博客](https://www.pi.website/blog/pistar06) |
| SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning | 21 位作者联合 | 2025 | [arXiv:2509.09674](https://arxiv.org/abs/2509.09674) |
| RLinf-VLA: A Unified and Efficient Framework for Reinforcement Learning of Vision-Language-Action Models | RLinf 团队 | 2025 | [arXiv:2510.06710](https://arxiv.org/abs/2510.06710)；[GitHub](https://github.com/RLinf/RLinf) |
| VLA-RFT: Vision-Language-Action Reinforcement Fine-tuning with Verified Rewards in World Simulators | 2025 团队 | 2025 | [arXiv:2510.00406](https://arxiv.org/abs/2510.00406) |
| π_RL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models | 2025 团队 | 2025 | [arXiv:2510.25889](https://arxiv.org/abs/2510.25889) |

注：未找到独立命名为"VLA-RL"的论文，与 SimpleVLA-RL / VLA-RFT / π_RL 等属于同一波"在线/离线 RL 微调 VLA"工作簇，**"VLA-RL"作为专有名词待核实**。

**核心方法与自动化/人工划分：**

- **RoboCat**：基于 Gato 架构的视觉目标条件决策 Transformer；自我改进循环为——用 100–1000 条人工遥操作演示微调出"分身 agent"→分身在该新任务/新机械臂上自主练习约 1 万次生成更多数据→喂回主模型再训练。*自动化*：策略自主练习、数据回收、主模型再训练；*人工*：初始 100–1000 条演示仍需人工遥操作，任务本身是人工预先设定好的，练习环境/场景也是人工搭建，没有失败诊断后修改任务这一步。
- **Self-Improving Embodied Foundation Models**（DeepMind 2025）：两阶段后训练——SFT（行为克隆+"剩余步数"预测）之后进入 Self-Improvement 阶段，用"剩余步数"预测自动导出稠密奖励函数和成功判别器，使机器人集群能在极少人工监督下自主练习新任务并推广到训练数据分布之外的技能。*自动化*：奖励函数提取、成功检测、自主练习、策略更新全部自动化，是目前"用自身经验持续改进策略"里自动化程度最高的工作之一。*人工*：任务集合、场景/资产、初始 SFT 数据依旧是人工提供；没有"自动发现新任务"或"自动修改仿真资产"的环节——它改进的是策略在已知任务分布内外推的能力，而不是任务/环境库本身。
- **π*0.6 / RECAP**（Physical Intelligence 2025）：RECAP = RL with Experience and Corrections via Advantage-conditioned Policies，三阶段——离线 RL 预训练→任务级示范微调→用真实环境中的"自主执行+专家实时纠正+奖励反馈"做在线 RL。真实结果：疑难任务吞吐量翻倍、失败率降低 2 倍以上；连续 18 小时无人干预做咖啡、叠 50 件新衣物、组装/贴标 59 个真实工厂纸箱。*自动化*：策略自主执行、奖励打分、策略更新；*人工*：任务本身固定（咖啡机、叠衣、装箱），专家纠正（intervention）仍由人类实时提供，是"人在环"式 RL 而非全自动，且不涉及任务/场景资产的自动变更。
- **SimpleVLA-RL / RLinf-VLA / VLA-RFT / π_RL**：2025 年井喷的"用 GRPO/PPO 等 RL 算法在仿真里对 VLA 做在线微调"的基础设施与算法类工作。SimpleVLA-RL 仅用单条轨迹+0/1 结果奖励在 LIBERO 等基准上把成功率推到 98.5%（+12%）；RLinf-VLA 提供统一异步 RL 训练框架，在 LIBERO/ManiSkill/RoboTwin 上分别达到 98.11%/97.66%/84.63%；VLA-RFT 用"世界模型模拟器"自动生成密集轨迹奖励，400 步微调即超过监督基线。*自动化*：rollout 采样、奖励计算、策略梯度更新全流程自动化。*人工*：任务集合、仿真环境/资产完全沿用既有基准（LIBERO/ManiSkill/RoboTwin 等）不做任何改变，人工仍需设计/选择奖励信号形式，这些工作的目标是"在同一任务分布内把成功率刷高"，不触碰任务生成或环境资产层。

**E 部分整体与目标闭环的差距**：这组"自我改进"工作的共同模式是——**固定任务/环境，只让策略通过自身 rollout 和 RL 信号变强**。没有一篇论文报告"根据 VLA 持续失败，自动诊断出是资产碰撞体缺失/专家脚本节奏不对/任务过难，然后自动新增或修改仿真任务、资产或专家脚本，再灌回训练"这样的反馈环。DeepMind 的 Self-Improving Embodied Foundation Models 和 π*0.6/RECAP 是目前"策略层自我改进"里自动化程度最高、也最接近生产部署的工作，但它们改进的对象始终是"策略参数"，而非"任务/仿真环境本体"——这正是与目标闭环最大的空白点：**目前没有证据表明任何已发表工作把"失败诊断"的输出接到了"仿真任务/资产/专家脚本的自动生成或修改"上**。

---

## F. 具身：自动评测与失败诊断

检索确认"Robo-Reflect"作为独立论文名称**未找到可靠的同名匹配**（待核实）；与之形近的是 RoboReflect（Grasping Ambiguous-Condition Objects, arXiv:2501.09307），但场景是抓取纠错而非通用失败诊断，仅作补充。

### F1. VLM/LLM 做失败检测与评测

| 论文 | 作者/机构 | 年份 | 链接 | 核心方法/自动化-人工/关键数字/差距 |
|---|---|---|---|---|
| **AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation** | NVIDIA (NVlabs) | 2024（ICLR 2025） | [arXiv:2410.00371](https://arxiv.org/abs/2410.00371)；[项目页](https://aha-vlm.github.io/) | 用 **FailGen** 框架程序化地对仿真成功演示做扰动，自动生成大规模"失败轨迹"数据集，微调专用失败检测/推理 VLM。扰动规则、失败分类体系由人工设计。比第二名 VLM 高 10.3%，比 6 个对比模型平均高 35.3%。差距：只做"检测+解释"，没有把失败原因接回"自动修改仿真任务/资产/专家"。 |
| **REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction**（含 RoboFail 数据集） | Zeyi Liu, Arpit Bahety, Shuran Song，Columbia | 2023 | [arXiv:2306.15724](https://arxiv.org/abs/2306.15724) | 把多模态执行记录汇总成分层文本摘要，LLM 做"渐进式"失败归因并驱动语言规划器生成修复计划；提出 RoboFail 数据集（100+失败案例）。任务/失败场景是人工预先设计并采集，失败标注含人工核验。差距：纠正停留在"当前 episode 内重规划再执行"，不涉及仿真任务/资产的持久修改，不产生新训练数据用于重训练策略。 |
| **FailBench: How Reliable are VLMs at Judging Robot Task Success?** | Zaruhi Navasardyan 等，Metric AI Lab | 2025 | [arXiv:2609.03611](https://arxiv.org/abs/2609.03611) | 汇总 14 个来源共 2197 条操作轨迹，系统评测 13 个 VLM-based 失败/成功判定器可靠性。最好模型（Gemini 系列）balanced accuracy 仅 **0.77**；GPT-4o 为 0.69；接触密集型装配任务上没有模型超过 0.60（接近随机）。差距：直接证明当前 VLM-critic 在接触/力相关任务上不可靠——若把"VLM 打分"接入自动闭环作为奖励信号或课程触发器，0.6-0.77 的准确率会引入噪声甚至误导后续的任务/专家修改。 |
| **Vision-Language Models for Robot Success Detection** | Fiona Luo 等 | 2024（AAAI） | [AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/30552) | 把成功检测建模为 VQA 问题，微调开源 VLM 做二分类成功判定。跨环境泛化差。提出"可作为奖励信号驱动策略改进"设想，但未实现"评测结果→修改仿真任务/资产/专家脚本"的具体机制，仍是概念性展望。 |

### F2. 自动课程生成（Automatic Curriculum）

| 论文 | 作者/机构 | 年份 | 链接 | 核心方法/自动化-人工/关键数字/差距 |
|---|---|---|---|---|
| **OMNI: Open-endedness via Models of human Notions of Interestingness** | Jenny Zhang, Joel Lehman, Kenneth Stanley, Jeff Clune 等 | 2023/2024 | [arXiv:2306.01711](https://arxiv.org/abs/2306.01711) | 用基础模型充当"人类兴趣度"代理模型，结合"学习进度"曲线调度任务采样。任务选择自动化；任务/环境参数空间仍人工预先定义。差距：只解决"选哪个已有任务练"，不解决"任务/资产/专家从哪来"。 |
| **OMNI-EPIC: ...Environments Programmed in Code** | Jenny Zhang, Shengran Hu, Cong Lu, Jeff Clune 等 | 2024 | [arXiv:2405.15568](https://arxiv.org/abs/2405.15568) | 让基础模型直接生成环境代码，以智能体过去成功/失败经验为"垫脚石"递进出难度合适、内容新颖的任务。一次运行完成 16 个任务、失败 6 个、1 个被判定"无趣"。**是与目标闭环最接近的通用工作之一**：用智能体的失败/成功历史驱动新环境代码生成。差距：（1）环境是抽象 RL 任务而非机器人真实资产/物理专家演示；（2）没有 VLA 训练与真实/仿真基准评测的接回；（3）没有"脚本专家生成演示数据"这一步。 |
| **Eurekaverse: Environment Curriculum Generation via Large Language Models** | Alan Luo, Jeff Clune, Sergey Levine 等 | 2024 | [arXiv:2411.01775](https://arxiv.org/abs/2411.01775) | LLM 生成"地形程序"，环境难度随策略能力共同进化，面向四足机器人跑酷。生成课程训练出的策略优于人工设计课程，并**成功迁移到真实机器人**（sim-to-real）。差距：面向关节控制型运动技能，不是操作类 VLA；没有"VLM 观察失败模式→归因→修改资产"的显式诊断环节，是隐式的（用成功率驱动难度）。 |
| **Voyager: An Open-Ended Embodied Agent with Large Language Models** | Guanzhi Wang 等，NVIDIA/Caltech/UT Austin/Stanford | 2023 | [arXiv:2305.16291](https://arxiv.org/abs/2305.16291) | 自动课程（根据当前状态/已完成/失败任务生成下一目标）+不断增长的技能库+迭代式代码生成（用报错和自我验证反复修正代码直到成功），全自动无需人工干预。获取新物品种类速度比 AutoGPT 快 3.3 倍，解锁科技树里程碑快 15.3 倍。差距：载体是 Minecraft 离散动作/代码原语，没有连续物理仿真、没有资产/任务的物理建模、没有真实 VLA 训练。 |

**F 部分小结**：失败观察/评测环节已能做到"自动打分/自动归因"，但 FailBench 表明当前 VLM-critic 在接触密集型任务上准确率仅约 0.6-0.77，**可靠性不足以直接作为自动化闭环的可信信号**。自动课程环节（OMNI-EPIC、Eurekaverse、Voyager）已经在各自领域做到"用智能体表现（含失败）自动生成下一批更合适的任务/环境"，是离目标闭环概念上最近的通用范式，但**没有一篇**把"VLM/LLM 判定的失败原因"显式转化为"对操作类仿真任务的资产摆放、脚本专家逻辑的具体修改"，再录制新演示数据重新训练 VLA 并评测。

---

## G. 空中操作/无人机操作的仿真基准与数据

**AM-Bench 的 arXiv 编号已核实为真实存在**（提交于 2026 年 9 月 1 日，注意这是未来批次的合理编号，非笔误）。

| # | 论文 | 作者/机构 | 年份 | 链接 |
|---|------|----------|------|------|
| 1 | AM-Bench: A Modular Simulation Suite and Benchmark for Aerial Manipulation Policy Learning | Yutong Wang, Dongjae Lee, Xiaofeng Guo, Yuanzhu Zhan, Yufei Jiang, Bavin Saravanan, Muqing Cao, Jia Xie, Chenyang Mao, Sebastian Scherer, Junyi Geng, Guanya Shi（CMU 机器人研究所为主，联合宾州州立、庆熙大学等） | 2026（CoRL 2026） | [arXiv:2609.00641](https://arxiv.org/abs/2609.00641)；[HTML](https://arxiv.org/html/2609.00641v1)；[项目页](https://ambench.github.io/)；[GitHub](https://github.com/ambench/ambench) |
| 2 | AIR-VLA: Vision-Language-Action Systems for Aerial Manipulation | 待核实 | 2026 | [arXiv:2601.21602](https://arxiv.org/abs/2601.21602) |
| 3 | Flying Hand: End-Effector-Centric Framework for Versatile Aerial Manipulation Teleoperation and Policy Learning | LeCAR Lab（Guanya Shi 组，CMU 相关） | 2025 | [arXiv:2504.10334](https://arxiv.org/abs/2504.10334) |
| 4 | π, But Make It Fly: Physics-Guided Transfer of VLA Models to Aerial Manipulation | 待核实 | 2026 | [arXiv:2603.25038](https://arxiv.org/abs/2603.25038) |

**1. AM-Bench** — 面向多旋翼空中操作的模块化仿真套件与基准，28 页，涵盖欠驱动/全驱动/超驱动三类机身构型，共 12 个任务（接触、搬运、受限交互三类），支持可配置气动扰动与执行器饱和，提供标准低层控制器和基线策略学习算法；通过三组仿真研究+真实世界验证+一次硬件学习管线测试展示诊断价值。*自动化*：任务/本体/扰动配置模块化可组合。*仍靠人*：12 个任务本身是人工设计，未见 LLM/生成模型自动提出新空中任务的机制。**这是一个标准化评测基准，而非自动数据/任务生成引擎**。差距：只解决"评测"这一环，完全没有失败驱动的任务/资产/专家自动生成，也没有配套自动演示数据生成流水线（对标地面领域的 MimicGen）。用户记忆笔记中提到的"手无碰撞、夹爪 reset、实心桶、力伺服、弹道专家"等资产 bug，未在公开摘要/搜索结果中找到对应描述，应为用户自己使用该代码库时发现的实践问题，非论文公开内容，**建议用户自行确认**。

**2. AIR-VLA** — 号称首个专为 VLA 模型设计的空中移动操作基准，指出地面基准因缺乏空中平台运动学设计和对应任务数据而不适用。方法细节、作者机构、自动化程度**待核实**（仅获得基本定位信息，检索中还发现可能存在后续版本 AIR-VLA+，arXiv:2606.12859，同样待核实）。

**3. Flying Hand** — 末端执行器为中心的空中操作遥操作/策略学习框架，高精度控制器让人类操作者采集高质量演示。*仍靠人*：**演示数据采集依赖人类遥操作**，不是自动生成。差距：说明空中操作领域数据采集自动化程度明显落后于地面机械臂领域，尚无类似 DexMimicGen/MimicGen 的"从少量演示自动泛化增广"方法。

**4. π, But Make It Fly** — 把地面 VLA 模型（如 π0 系列）通过物理引导方式迁移到空中操作场景，属"模型迁移"而非"数据/任务自动生成"。细节和数字待核实。

**核心结论——空中操作领域是否存在自动化数据引擎（类比 MimicGen/DexMimicGen）：未检索到。** 多组关键词搜索均未发现任何论文为空中操作构建"从少量人类演示自动生成大规模多样化演示数据"的系统。当前数据来源主要是：(a) 人工遥操作（Flying Hand）；(b) 传统轨迹优化/最优控制生成专家轨迹；(c) 少量 RL 训练策略。AM-Bench 提供的是标准化评测基准与基线算法，不是自动数据生成引擎。**这是本次调研中明确的空白点：空中操作领域缺乏"自动任务生成+自动专家/演示生成+自动评测"的数据飞轮**，无论是对标 GenSim/RoboGen 的"批量开环生成"层面，还是更进一步的"失败驱动自动闭环"层面，均处于起步阶段；AM-Bench（2026）只填补了评测协议这一个环节的空白。

---

## 综合分析一：闭环各环节自动化程度矩阵

五个环节：①失败观察（诊断哪里错了）；②任务/资产/专家修改（生成或改写仿真任务、场景资产、脚本专家/奖励）；③数据录制（采集/合成演示数据）；④训练（模型参数更新）；⑤评测（衡量是否变好）。

| 代表工作 | ①失败观察 | ②任务/资产/专家修改 | ③数据录制 | ④训练 | ⑤评测 | 备注 |
|---|:---:|:---:|:---:|:---:|:---:|---|
| RoboGen (B) | ✗ | ✓（自动，开环） | ✓（自动） | ✓ | ✗（弱） | 生成驱动力是覆盖多样性，非失败反馈 |
| GenSim / GenSim2 (B) | ✗ | ✓（自动，开环） | ✓（自动） | ✓ | ✓（同分布） | 广度优先，不针对特定失败 |
| BBSEA (B) | △（粗粒度执行反馈） | ✓（任务提议） | ✓ | ✓ | △ | 唯一有反馈环但粒度粗、规模小 |
| EmbodiedGen V2 / SAGE (B) | ✗ | ✓（agentic 自我修正） | — | — | ✗ | 触发是人类意图，非模型失败；critic 判合理性非策略表现 |
| Eureka / DrEureka / Text2Reward (C) | △（RL 训练曲线） | ✓（奖励/专家代码自动重写） | —（RL 非模仿学习） | ✓（RL） | ✓（RL 回报） | 反馈信号是奖励收敛情况，不是 VLA 失败案例 |
| Code as Policies / VoxPoser (C) | ✗ | ✓（零样本生成策略代码） | ✗ | ✗ | 人工观察 | 直接执行，不产出训练数据 |
| MimicGen / DexMimicGen / SkillMimicGen (D) | ✗ | ✗（任务/资产固定） | ✓（自动增广） | ✓ | ✓ | 只做同任务内数据扩增 |
| RoboCasa (D) | ✗ | ✓（场景/资产生成，一次性） | ✓（自动增广） | ✓ | ✓ | 生成后任务集合固定，不再迭代 |
| AutoRT / DIAL (D) | ✗ | ✗ | ✓（自动/半自动采集） | ✓ | 人工/弱 | 任务提议基于场景感知，非失败诊断 |
| DexFlyWheel (D) | ✗（隐式，靠成功率） | ✗（任务固定） | ✓（rollout 自动采集） | ✓ | ✓ | 飞轮转的是策略-数据，非任务本身 |
| RoboCat / Self-Improving Embodied FM / π*0.6-RECAP (E) | ✗（策略层面，无诊断） | ✗ | ✓（自主 rollout） | ✓ | ✓ | 固定任务，只让策略变强 |
| SimpleVLA-RL / RLinf-VLA / VLA-RFT / π_RL (E) | ✗ | ✗ | ✓（RL rollout） | ✓ | ✓（基准内） | 环境/任务不变，只刷同分布成功率 |
| AHA / REFLECT / FailBench (F1) | ✓（核心贡献） | ✗ | ✗ | ✗ | ✓（作为评测/critic） | 止步于"打分/解释"，不回接生成 |
| OMNI-EPIC (F2) | ✓（成功/失败历史驱动） | ✓（生成环境代码） | ✗（非物理机器人演示） | ✗（该工作本身不含 VLA） | ✓（自评趣味性/可学习性） | **通用领域内最接近完整闭环**，但非具身操作物理任务 |
| Eurekaverse (F2) | △（隐式，靠成功率） | ✓（地形代码生成） | ✓（RL 训练数据） | ✓ | ✓（含 sim-to-real） | 面向运动而非操作，无显式失败归因 |
| Voyager (F2) | ✓（失败/成功驱动课程） | ✓（技能库+课程） | —（游戏内非物理演示） | — | ✓ | 非物理仿真、非真实 VLA |
| AM-Bench (G) | ✗ | ✗ | ✗ | ✗ | ✓（标准化基准） | 只覆盖评测这一环 |

**矩阵结论**：
- 覆盖①（失败观察/诊断）且做得可靠的工作（AHA、REFLECT、FailBench）几乎都不触及②（任务/资产修改）；FailBench 更直接证明当前 VLM-critic 准确率（0.6–0.77）尚不足以支撑高置信度的自动闭环。
- 覆盖②（任务/专家自动修改）的工作（RoboGen、GenSim 系列、Eureka 系列、OMNI-EPIC、Eurekaverse）绝大多数是**开环生成**（覆盖多样性驱动）或**用聚合指标（成功率/回报）隐式驱动**，只有 BBSEA 和 OMNI-EPIC 做到了显式"用智能体历史表现（含失败）驱动生成"，但前者规模小、粒度粗，后者不涉及物理机器人操作和真实数据录制。
- 覆盖③④⑤（数据、训练、评测）三环的工作最多、最成熟（MimicGen 系列、RoboCasa、E 部分全部工作），但都建立在**任务/环境固定不变**的前提上。
- **没有一项工作五个环节全部覆盖，且以"具体到某个仿真任务的 VLA 失败案例"为触发信号**——这正是下一节要展开的空白。

---

## 综合分析二：空白与机会

### 1. 核心空白：没有工作把"VLA 失败"接回"仿真任务/资产/专家的自动修改"

综合 A–G 全部检索结果，**没有找到任何已发表工作同时满足**：(a) 以具体机器人策略（尤其是 VLA）在某个仿真基准上的**失败案例**为输入，(b) 用 VLM/LLM **自动诊断**失败原因（资产碰撞体缺陷、专家脚本时序错误、任务难度/物理参数不合理等具体层面，而非笼统的"失败/成功"标签），(c) 据此**自动生成或修改**仿真任务定义、场景资产、脚本专家/奖励代码，(d) 自动**录制**新演示数据，(e) 重新**训练** VLA 并**评测**，形成持续迭代。

现有工作簇的模式可归纳为三类，各自解决了闭环的一部分，但没有人把它们串起来：
- **"批量造任务"派**（RoboGen、GenSim/GenSim2、Gen2Sim、RoboCasa、EmbodiedGen、SAGE）：生成开环、覆盖导向，驱动力是"多样性"而非"哪里失败"。
- **"策略自我改进"派**（RoboCat、DeepMind Self-Improving Embodied FM、π*0.6/RECAP、SimpleVLA-RL 系列）：任务/环境固定，只让策略参数变强，是"同一个任务练得更好"而不是"任务库本身该怎么长大"。
- **"失败诊断+课程"派**（AHA、REFLECT、FailBench、OMNI-EPIC、Eurekaverse、Voyager）：在各自的窄场景里做到了"观察失败→影响后续任务生成"，其中 OMNI-EPIC 和 Voyager 在抽象 RL/游戏环境中已经跑通了完整雏形，但都不是"物理机器人操作任务+VLA 训练"的场景；Eurekaverse 是物理场景（四足运动）但驱动信号是聚合成功率而非可解释的失败归因。

**最接近题目描述闭环、但仍不完整的两个候选**：
- **OMNI-EPIC**（arXiv:2405.15568）：机制上"用智能体成功/失败历史驱动 LLM 生成新环境代码"与题目描述高度同构，是目前通用 AI4AI/开放式学习领域里形态最完整的雏形，但环境是抽象 RL/游戏式任务，不产生用于 VLA 模仿学习的演示数据，也没有资产级别的物理仿真修改。
- **BBSEA**（arXiv:2402.08212）：在具身操作场景里做到了执行反馈回传影响任务提议，但规模小、反馈粒度粗（"这个任务能不能做"而非"错在哪个具体环节"），未涉及资产/专家脚本的自动修改，也没有 VLA 训练闭环。

**若要真正落地题目描述的闭环，目前文献中缺失的关键拼图包括**：
1. **可靠的细粒度失败归因器**：FailBench 已证明现有 VLM-critic 在接触密集任务上准确率仅 0.6–0.77，需要比"成功/失败"更细的归因（如"因为夹爪碰撞体缺失导致穿模"），这类专用诊断模型/流程目前只有 AHA 做了初步尝试（且是仿真扰动生成的失败数据，非真实 VLA 部署产生的失败）。
2. **失败→资产/任务代码的自动映射器**：需要把①的诊断结果转成对仿真资产 URDF/碰撞体、任务 YAML/DSL、脚本专家 Python 代码的具体编辑操作，这一步在检索范围内**完全没有先例**——现有"LLM 写仿真代码"工作（RoboGen、Eureka、OMNI-EPIC）都是从零生成新任务/奖励，而不是"针对已有任务的已知 bug 做定点修复"。
3. **端到端验证的产线**：即使某个环节各自存在（如 C 部分的专家代码自动生成+D 部分的 MimicGen 数据增广+E 部分的 RL 微调），也从未有人把它们串成"资产 bug 修复→重新录制→重新训练 VLA→重新在同一仿真基准评测→确认失败模式消失"的完整实验并公开报告结果。

### 2. 空中操作（aerial manipulation）领域的具体空白

相比地面机械臂领域，空中操作的自动化程度全面落后：
- **无自动任务/场景生成引擎**：地面有 RoboGen/GenSim/RoboCasa，空中操作目前只有人工设计的固定任务集合（AM-Bench 12 个任务）。
- **无自动数据增广引擎**：地面有 MimicGen/DexMimicGen/SkillMimicGen，空中操作的演示数据仍主要依赖人类遥操作（如 Flying Hand）或传统轨迹优化，未见"从少量演示自动泛化生成大规模多样化空中操作演示"的系统。
- **无自动奖励/专家生成**：Eureka/DrEureka 类工作尚未见到迁移到空中操作（欠驱动/全驱动机身、气动扰动、执行器饱和等空中特有物理约束会让 LLM 生成奖励/专家的难度显著提高，这本身可能是一个有价值的研究方向）。
- **评测基础设施刚起步**：AM-Bench（2026）刚刚提供了标准化的模块化仿真基准，这是自动化数据引擎的**必要前提**（有了标准接口才能让 LLM/Agent 批量生成任务变体），但目前尚无人在此基础上搭建自动化闭环。
- **机会**：AM-Bench 的模块化设计（可配置本体/扰动/任务）恰好为"LLM/Agent 自动生成任务变体+自动诊断失败+自动修专家"提供了理想的实验平台——这是一个明确的、几乎空白的研究机会，尤其是把 C 部分（Eureka 类自动专家生成）和 F 部分（OMNI-EPIC 类失败驱动课程）的方法论迁移到空中操作场景，同时补上"细粒度失败归因"和"失败到仿真代码编辑的自动映射"这两个目前全领域都缺失的环节。

### 3. 对本项目闭环的启示

- 可直接复用的成熟组件：**任务/场景生成**用 EmbodiedGen/RoboGen 思路，**专家/奖励生成**用 Eureka/DrEureka 思路，**数据增广**用 MimicGen/DexMimicGen 思路，**策略侧自我改进**用 π*0.6/RECAP 或 SimpleVLA-RL 类在线 RL 思路。
- 需要自研、文献中没有先例的核心拼图：①面向仿真资产/脚本代码级别的**细粒度失败归因**（不满足于"成功/失败"二分类，需要 FailBench 揭示的可靠性问题被解决）；②**失败归因到仿真代码编辑的自动映射器**（目前最大空白）；③把以上环节和 VLA 训练/评测串成可重复验证的**完整闭环实验**并度量迭代收益。这三点，加上"空中操作"这个具体领域几乎全空白的现状，共同构成了本项目相对于现有文献的主要创新空间。

---

## 附：待核实条目清单

- Genesis 物理引擎是否有正式配套 arXiv 研究论文（仅找到 GitHub/项目页）。
- SAGE（arXiv:2602.10116）的作者/机构、关键量化结果。
- AIR-VLA（arXiv:2601.21602）、AIR-VLA+（arXiv:2606.12859）、"π, But Make It Fly"（arXiv:2603.25038）的作者机构与详细方法/数字（仅获得基本定位信息）。
- VoxPoser 相对基线的具体量化对比数字。
- "Robo-Reflect"作为独立论文名称未找到可靠匹配，与 RoboReflect（arXiv:2501.09307，聚焦模糊条件抓取纠错）不完全等同。
- "VLA-RL"作为专有论文名称未找到，可能与 SimpleVLA-RL/VLA-RFT/π_RL 等混淆。
- 用户记忆笔记中提到的 AM-Bench"手无碰撞、夹爪 reset、实心桶、力伺服、弹道专家"等资产 bug，未在 AM-Bench 论文公开摘要中找到对应描述，应为用户自己使用该代码库/资产时的实践发现，非论文正文内容。
