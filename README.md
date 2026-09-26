# STTW_CONTROL / Isaac Sim sim2sim 迁移

把在 **MuJoCo MJX-Warp** 上训练的策略迁移到 **Isaac Sim / PhysX** 运行，并验证两者是否真的等价。

**当前结论：sim2sim 尚未通过。** 所有候选策略在 PhysX 下的物理测试全部失败。仓库里保留的是完整的诊断过程、可复现的失败、以及已被证据支持/否定的各项假设——不是一份成功报告。

本仓库是 `/home/qy/ISAAC——SIM` 独立工作根目录的归档，源码与产物原样搬运，未作修改。原始 ML 训练环境（DVGC / STTW_CONTROL / JIT）不在本仓库，且未被改动。

---

## 一、要解决的问题

任务：一个 6 刚体轮足机器人（自由根 + 5 铰链，前轮被动、转向/后轮/髋/膝主动，76 维三帧历史观测、20 ms 控制周期）完成起跳。

源端 `transition_4988928` 的 Actor 在 MJX-Warp 里可用。把它搬到 Isaac Sim 后：

- 模型导入成功、质量/惯量/关节限位/增益/力矩上限逐项核对一致；
- 但策略**每次都在 2 秒左右侧倾或偏航超限失败**。

问题因此从「能不能导入」变成「**两个引擎的物理到底哪里不一样**」。整个仓库的后续工作都是在回答这一句。

> 一条贯穿始终的纪律：不把导入成功当迁移成功，也不拿奖励排名当 sim2sim 成功。

## 二、项目走过的阶段

按时间顺序，每一阶段都有独立的入口报告和原始证据。**后一阶段会推翻前一阶段的乐观结论，这些推翻过程本身也保留在仓库里。**

| 阶段 | 日期 | 做了什么 | 结论 | 入口 |
|---|---|---|---|---|
| 0 环境与导入 | 2026-09-20 前 | Isaac Sim 5.1 环境、USD 构建、smoke、策略导出与推理对照 | 运行时可行 | [domain_audit](results/domain_audit/pre_audit_INDEX.md) |
| 1 sim2sim 物理审计 + 域随机化 | 2026-09-20 | 全参数审计、轮胎碰撞修正、两轮 DR 训练 | **原策略与 DR 候选 0/16 通过** | [sim2sim_report](results/sim2sim_report/INDEX.md) |
| 2 执行器响应诊断与参数标定 | 2026-09-20 | 20 组恒力矩 + 34 组驱动参数搜索 + 12 组独立验证 | 质量/惯量/关节映射在自由运动下数值一致（相对差 0.003%）；但**发现 5 ms RK4 的角度与步末速度严重偏离** | [actuator_report](results/actuator_report/INDEX.md) |
| 3 接触能力审计 | 2026-09-21 | 六组滑动测试 | PhysX 标量摩擦**无法表达**源端双切向摩擦 | [contact_capability](results/contact_capability_20260921/INDEX.md) |
| 4 原生 PhysX 1000 万步训练 | 2026-09-21~22 | 从原 Actor 初始化，重建 PPO，全采样在 PhysX | 训练跑完，**best 8 回合复测全部提前失败（2.02–3.72 s）** | [phase_u_10m](runs/physx_phase_u_10m_20260921/INDEX.md) |
| 5 训练契约移植审计 | 2026-09-26 | 把原 MJX/Brax 训练契约迁到 PhysX | 契约可移植、奖励数值复算通过；但**接口相同不等于输入相同** | [ppo_port_audit](results/ppo_port_audit_20260926/INDEX.md) |
| 6 迁移失败诊断（角速度） | 2026-09-26 | 公开跨引擎实现对照 + 真实 MJX-Warp/PhysX 逐通道比对 | **找到 RK4 sensor 采样阶段差异**；4 组驱动配置真实短测仍全部失败 | [migration_audit](results/migration_audit_20260926/INDEX.md) |
| 7 同输入响应辨识 | 2026-09-26（最新） | 向两引擎施加相同 SI 目标，比较完整响应 | 进行中 | [input_response](results/input_response_20260926/) |

## 三、当前卡在哪里

已用证据**确证**的差异（都不是靠视频推断）：

1. **传感器采样阶段与最终状态不同步。** 同一模型、初态、零动作，5 ms × 4 步后：真实 CUDA MJX-Warp 的机体系 Y 轴 gyro 读到 `−1.651473 rad/s`，而最终 `qvel` 是 `+0.339276 rad/s`；CPU MuJoCo 读到 `+0.334790`。原因是 Warp RK4 内部重算 sensor 后，最终加权积分没有再刷新，留下最后一个内部试探态读数。源 Actor 读 `gyro_local`、奖励读 `ang_global`，所以这项差异**确实同时进入了策略输入和奖励**。
   → 只替换构造诊断帧的 gyro，冻结 Actor 的后轮目标就变化 `+0.293565 rad/s`。

2. **粗步长强伺服下角度与步末速度不一致。** 孤立原后轮（I=0.0004165 kg·m²，速度增益 5，力矩上限 ±6 Nm，目标 12 rad/s），20 ms 后：源端 RK4 5 ms 得到 `q≈0.240096 rad, qd≈0`；PhysX 5 ms 得到 `q≈0.239000 rad, qd≈12 rad/s`；源端细化到 0.1 ms 后 `qd` 回到 12。5 ms 比该伺服线性时间常数 `I/k≈0.0833 ms` 大约 60 倍，饱和力矩在 RK4 内部阶段翻转。
   → **轮子确实在转，是返回的步末速度接近零**；这也解释了为什么画面相似会掩盖策略输入差异。

3. **PhysX 标量摩擦无法表达源端各向异性接触。** 源端轮地是 condim=6、双切向 `[5, 0.5]`；PhysX 当前只保留单一标量。当前名义配置用 5，**不能代表原 5/0.5**。

仍未隔离贡献度的因素：各向异性接触近似、原生加速度与末子步差分加速度不同、隐式/显式/RK4 执行器响应差异、接触 reset 缓存、历史 PPO 实现差异。**没有把任何一项称为唯一根因。**

明确**不能**做的事（已被否定的捷径）：给原策略加固定负号/比例/延迟来「修正」角速度；只调 PD 增益就宣称解释全部失败；用视频相似度判断动力学等价。

详见 [migration_audit/INDEX.md](results/migration_audit_20260926/INDEX.md)、[angular_audit.md](results/migration_audit_20260926/angular_audit.md)、[upstream_research.md](results/migration_audit_20260926/upstream_research.md)。

## 四、当前活跃的实验

`ACTIVE_TRAINING.json` 是唯一权威的运行入口记录。

- **fresh 从零重训**：384 环境、64 步序列、24 minibatch × 8 轮，随机初始化（未加载预训练权重），预算 9,977,856 转移，TensorBoard 端口 6007。
- 该运行在 73,728 步时进程退出（**原因未知，未编造**），已从 checkpoint 恢复，独立 runtime、独立 systemd 用户服务；恢复声明明确为**非逐位续接**。
- 本轮新增的角速度 opt-in 选项**没有接入**该恢复训练，原采样契约保持不变。

`runs/` 下每轮训练只保留轮次级元数据（`INDEX.md` / `status.json` / `learner_identity.json` / `runtime_audit.json`）；checkpoint、逐 transition 的 `steps.json`、`episodes.jsonl`、TensorBoard 事件文件**不在本仓库**。

## 五、仓库导航

```
├── README.md               ← 本文件
├── PROJECT.md              ← 详细任务叙事与每轮声明（最新在前）
├── INDEX.md                ← 按阶段的时间线索引
├── ACTIVE_TRAINING.json    ← 当前活跃训练的运行身份与进程核验
│
├── source_task.py          ← 源端任务定义（含 opt-in 的 angular_velocity_mode）
├── source_reset.py         ← 源端 reset 分布
├── usd_model.py            ← MuJoCo → USD 构建
├── physx_task.py           ← PhysX 侧任务实现
├── brax_physx_learner.py   ← PPO learner（PhysX 采样）
├── response_protocol.py    ← 阶段 7 的不可变同输入协议
├── probe_mjx_response.py   ← 在真实 MJX-Warp 上施加协议输入
├── probe_physx_response.py ← 在真实 PhysX 上施加协议输入
├── analyze_response.py     ← 联合 q/qd/omega 评分，保留负结果
├── drive_response_adapter.py ← 离散执行器响应适配候选
│
├── configs/                ← 域随机化、角速度 opt-in、目标测试声明
├── model/                  ← 源 MJD/XML + 构建后的 USD + STL
├── policy/                 ← Actor 权重的身份与推理验证记录
├── setup/                  ← 安装过程、依赖快照、设计文档、历次 smoke 日志
├── tests/                  ← 接口契约、奖励重建、重复性测试
├── results/                ← 各阶段报告 + 图 + 原始证据（见下）
└── runs/                   ← 各轮训练的轮次级元数据与报告
```

`results/` 中每个阶段目录通常包含：`INDEX.md`（结论与证据边界）、`figures/`（图，PNG/PDF）、原始 `*.json` / `*.csv`（可复算的中间证据）、以及被冻结的实现快照。**原始失败记录一律不覆盖、不删除。**

## 六、阅读建议

- **只想了解思路** → 本 README + [migration_audit/INDEX.md](results/migration_audit_20260926/INDEX.md) 的第 1、2、5 节。
- **想看完整时间线** → [INDEX.md](INDEX.md)，再看各阶段 `INDEX.md`。
- **想复现结论** → 每个 `INDEX.md` 都列出对应 CSV/JSON/PNG 与事前声明（`*_declaration.json`），以及该轮消耗的物理子步预算。
- **想知道哪些说法被撤回** → 各阶段报告的「哪些已验证，哪些没有」小节；`results/before_render_sync_audit/`、`results/explicit_servo_diagnostic/` 保留了被后续结论取代的中间状态。

## 七、工程约定

这些约定贯穿所有阶段报告，读的时候可以按它们校准对每句话的信任度：

- **事前声明**：每轮实验启动前先写死预算、停机条件、验收门槛（`drive_declaration.json`、`final_target_test.json`、`declaration.json` 等），事后不调整。
- **真实失败**：首个真实 `done` 或时间上限即停，失败后不延长轨迹、不继续模拟倒地杂耍。
- **声明适用范围**：每个结论都标注它覆盖什么、不覆盖什么（例如「不证明全状态或接触一致」「重复基线不算额外独立样本」）。
- **不伪造**：缺失的历史进程记录、未知的退出原因、无法补造的历史包身份，一律明确写「未知」，不做推断性填补。
- **物理预算计数**：每轮报告都给出实际消耗的物理子步数与上限。
