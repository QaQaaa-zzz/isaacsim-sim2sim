# 当前：迁移失败诊断与角速度适配（2026-09-26）

已核查公开跨引擎仓库并完成真实MJX-Warp/PhysX工程对照。发现RK4传感器采样阶段与最终qvel存在显著差异；当前world/body和rad/s转换没有发现错误。四种驱动配置均失败，未完成sim2sim。已加入显式末步角速度采样选项，默认历史行为保留；没有重启训练。

[本轮报告、图与原始证据](results/migration_audit_20260926/INDEX.md) · [角速度实测](results/migration_audit_20260926/angular_audit.md) · [上游仓库逐项对照](results/migration_audit_20260926/upstream_research.md) · [当前实施状态](PROJECT.md)

**最新现场状态：** 14:41旧fresh进程缺失的记录保留；另一个恢复流程已于14:45从73,728步checkpoint续起，当前记录5次更新、126,336累计采样转移，训练/评估/TensorBoard进程均存在，尚未完成。本轮新角速度选项未接入恢复训练。[最新核验](results/migration_audit_20260926/training_liveness_final.json)

---

# 当前：PhysX 从零重新训练

2026-09-26用户明确要求从零训练。此前恢复预训练权重的续训已停止。新一轮Actor/Critic随机初始化，归一化和Adam从零开始，384环境，独立预算9,977,856步；未加载指定checkpoint权重，只沿用其配置。

[新运行入口](runs/physx_source_ppo_fresh_10m_20260926/INDEX.md) · [实时状态](runs/physx_source_ppo_fresh_10m_20260926/status.json) · [初始化身份](runs/physx_source_ppo_fresh_10m_20260926/learner_identity.json) · [TensorBoard：fresh](http://127.0.0.1:6007/) · [初始化验证](results/ppo_port_audit_20260926/fresh_initialization_verification.json)

下方是历史运行快照；当前唯一活动入口以ACTIVE_TRAINING.json为准。

# DVGC → Isaac Sim 当前交付

**当前（2026-09-26）：已启动原MJX/Brax训练契约迁移后的新一轮PhysX训练。** 384环境、64步序列、24个minibatch×8轮；恢复指定源Actor、106维Critic及完整归一化统计，奖励保持不变。预算9,977,856个新转移，工程验证另计；尚无本轮能力提升结论。

[新运行入口](runs/physx_source_ppo_10m_20260926/INDEX.md) · [实时状态](runs/physx_source_ppo_10m_20260926/status.json) · [best](runs/physx_source_ppo_10m_20260926/best_model.json) · [TensorBoard](http://127.0.0.1:6007/) · [上轮失败原因及完整移植对照](results/ppo_port_audit_20260926/INDEX.md)

上轮已完成10,000,000步，best重测八回合全部失败，不能将奖励排名解释为sim2sim成功。[已交付视频、轨迹、关节与指令诊断](runs/physx_phase_u_10m_20260921/review/INDEX.md)。轮地仍是前向.5近似，侧向5未保存；物理与加速度观测差异、重复评分变化均列入本轮声明。

以下为先前阶段记录，旧文中的“最新”“尚未启动”只对应当时日期；当前状态以以上新运行和ACTIVE_TRAINING.json为准。

**最新：已启动原生 PhysX 的 1000 万步训练流水线。** 根据用户新指令，轮地仅匹配初态前行方向，使用各向同性0.5和候选PD。任务未完成，实时进度以状态文件为准。[运行入口](runs/physx_phase_u_10m_20260921/INDEX.md) · [状态](runs/physx_phase_u_10m_20260921/status.json) · [best](runs/physx_phase_u_10m_20260921/best_model.json) · [TensorBoard](http://127.0.0.1:6007)。以下“尚未启动”均为此前条件下的历史记录。

**本轮进展：候选 PD 已配置并在 PhysX 实测验证。** 六组滑动测试确认当前原生材质无法表达源端双切向摩擦；需接触求解扩展，1000 万步训练尚未启动。[实现与摩擦能力证据](results/contact_capability_20260921/INDEX.md)。

**2026-09-21：1000 万步训练尚未启动。** 本次实际 PhysX 核对：质量和默认 PD 数值一致，轮地摩擦不一致，未满足“相同才开始”的条件。[核对与接口缺口](results/training_preflight_20260921/INDEX.md)。

**最新：已完成无接触执行器诊断及一轮 Isaac 参数标定。** 20 组恒力矩测试最大关节速度相对差约 0.003%；发现原始 5 ms RK4/MJX 的角度变化与步末速度严重偏离，原生 MJX 已复核。34 组驱动参数搜索、12 组独立输入验证后，指定原策略在 PhysX 的名义失败由 0.32 s 延后到 0.96 s，但仍侧倾失败、未完成起跳。候选未通过迁移验收，本轮无训练。

[最新诊断与标定报告](results/actuator_report/INDEX.md) · [原参数/标定参数视频](results/actuator_report/comparison.mp4) · [角度与速度关键证据](results/actuator_report/rotor_state_channels.png) · [验证结果](results/actuator_report/verification.json)

**此前阶段，2026-09-20：sim2sim验收未通过。** 已修正轮胎碰撞近似并完成物理参数审计、两轮域随机化训练与独立测试；原策略和DR候选在PhysX的16个声明测试抽样中均0/16通过。未替换原checkpoint。

[完整参数审计与结果](results/sim2sim_report/INDEX.md) · [四画面对照视频](results/sim2sim_media/comparison.mp4) · [名义对照图](results/sim2sim_report/nominal_comparison.png) · [全部案例CSV](results/sim2sim_report/all_cases.csv)

质量、惯量、关节限位/增益/力矩上限已核对；**摩擦、接触柔顺性和离散驱动响应不一致**。具体数值和失败证据见报告，不能再把导入成功当迁移成功。

独立根目录 `/home/qy/ISAAC——SIM`；环境 `.venv`；Isaac Sim5.1/Python3.11，原MuJoCo环境和DVGC/STTW源码不改。当前入口在原任务终止条件触发时停止：

```bash
cd '/home/qy/ISAAC——SIM'
./run_policy.sh
```

该命令运行原 `transition_4988928`，是可复现失败诊断，不是成功演示。

[当前随机化配置](configs/domain_randomization.json) · [第二轮训练声明](results/dr_solver_training/declaration.json) · [源端筛选](results/dr_solver_development/selection.json) · [目标端测试声明](configs/final_target_test.json) · [依赖快照](setup/requirements-current.freeze.txt)

[历史环境选择、安装与最初smoke记录](results/domain_audit/pre_audit_INDEX.md)。旧文档是历史快照，旧视频及失败日志未覆盖。
