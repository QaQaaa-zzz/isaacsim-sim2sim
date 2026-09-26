# MJX → PhysX 适配当前任务（2026-09-26）

用户要求查找并学习公开跨仿真器实现，对照本地移植，优先核实角速度差异并继续适配。

## 范围与验证计划
- [x] 读取上游原始源码与官方接口，明确实际迁移方向、数据语义和可复用方案。
- [x] 追踪根/关节角速度、坐标系、四元数、COM/根原点、传感器采样时刻与驱动离散响应。
- [x] 复核已保存失败轨迹和当前训练身份，保留全部历史负结果。
- [x] 先建立可失败的接口数值测试，再对确认问题作最小修复；在独立输出目录进行有界真实 PhysX 验证。
- [x] 更新审计报告、复现实验入口、当前状态与限制。

## 本轮声明
工作只写 /home/qy/ISAAC——SIM；/home/qy/DVGC 与 STTW_CONTROL 源码只读。当前目标根目录没有 Git 元数据、AGENTS.md 或既有 PROJECT.md，不能编造 commit/push。
14:41核验fresh旧进程已退出；最后复查另一个流程已从73,728步恢复并使用独立runtime，本轮不操作该训练；新增适配经短测试后明确标注适用范围。
优先执行无动力学测试及单环境诊断；工程验证预算最多 20,000 个物理子步，非训练/性能结论。停机条件为非有限状态、接口契约断言失败或应用异常。诊断产物输出 results/migration_audit_20260926，原记录不覆盖。
初步比较方案：直接替换全框架成本高且改变过多；只调PD不能区分接口/动力学；本轮采用上游语义契约与单变量探针，确认问题后原位修复。

## 已核实现状
旧 10M PhysX 训练完成，best重测 8/8 失败；fresh旧进程缺失记录保留；现在另一个恢复流程已运行，见training_liveness_final.json。先前增益拟合没有同时对齐 q/qd，摩擦方向差异仍存在，不能因发现一个问题宣称解释全部失败。

## 进展：上游契约和角速度
公开参考已固定到commit，详见results/migration_audit_20260926/upstream_research.md。已确认运行时root角速度为world rad/s，Actor/critic需转body；USD文件角度单位不能混入runtime。新增MuJoCo同步传感器独立oracle测试覆盖单位姿态、90度yaw、复合姿态、乱序关节及Euler导数区别，5项相关测试通过。

当前源路径MuJoCo3.6.0的RK4采样时刻已用真实MJX-Warp核查，CPU/JAX结果不可替代Warp结论；冻结源包身份仍有边界。复读原轨迹确认孤立后轮20ms：源RK4 5ms的q=0.240096rad、qd=0，PhysX q=0.239000rad、qd≈12rad/s；源细化0.1ms后qd=12。此为离散执行器差异，不是单位换算。

有界适配声明见drive_declaration.json：当前hybrid栈1ms物理/20ms控制，原transition_4988928 Actor、单地面初态；当前拟合增益×2重复、只恢复源后轮增益、只恢复源转向增益、两者恢复。每例首个真实done或2秒截止，总上限10,000子步，不训练、不改在训配置，不把候选当合格模型。

## 用户授权恢复训练（2026-09-26）
用户报告fresh仿真退出并要求继续训练。恢复同一fresh实验的73,728步checkpoint，保留Adam/统计/采样RNG、原best和9,977,856总预算。PhysX场景重建并声明非逐位续接；冻结原训练runtime及资产，不混入上文独立迁移探针。恢复记录：runs/physx_source_ppo_fresh_10m_20260926/recovery/attempt_0001/INDEX.md。独立systemd用户服务管理训练与TensorBoard，退出原因尚未确定。13项恢复/learner测试通过，待核实恢复后的第4批进展。

## 2026-09-26 新实测与适配决定
- 上游仓库/失败审计已完成；当前源路径实际是MuJoCo/MJX3.6.0，隔离CPU是3.3.7，历史包身份不能凭路径补造。
- 实际Warp零动作4子步发现sensor保留RK4试探态：20ms body gyro y=-1.651473，而最终qvel y=+0.339276 rad/s；CPU/Warp最终状态近似。三条轨迹12子步，详情angular_audit.md。
- 已完成四驱动配置及baseline重复：76/80/93/91/76控制步，全部真实失败；8320 rollout+20初始化=8340 PhysX子步。基线重复逐值一致。不采用任何候选，不扩大训练。
- 已完成2次纯离线Actor敏感性检查；只替换gyro会改变动作，但不构成闭环失败归因或改善证明。
- 根据上游统一状态契约增加单一opt-in适配：隔离source_task.py的angular_velocity_mode=end_state使Actor与奖励共用最终qvel角速度；默认native_sensor保持历史数值。仅修采样契约，物理/加速度/奖励公式不改，原代码备份在implementation_before。不会把新模式当原checkpoint等价协议。
- 图、CSV与重复性：results/migration_audit_20260926/figures。相关测试29项与新增source opt-in的5项均通过；最后独立复核33 passed/1 deselected，默认兼容测试不重复消耗物理预算。

恢复验证完成：第4批、98,304转移、768次Adam，checkpoint保存及开发评分通过，TensorBoard新标量已核实。服务仍运行；训练总预算尚未完成。

## 本轮交付完成
报告入口：[迁移审计与适配](results/migration_audit_20260926/INDEX.md)。独立审查无阻断；本轮工程流程完成，但sim2sim成功仍未验证，四种驱动配置全部失败，不增加正式训练。新角速度配置configs/source_end_state_angular.json仅供显式新协议使用。最后核验另一个恢复流程已启动，ACTIVE记录当前存活状态并保留先前进程退出证据；本轮opt-in不改变恢复训练。Git仓库和远端均缺失，本轮无commit/push。

## 2026-09-26 用户指定：同输入响应辨识与匹配

目标：直接向真实 MJX-Warp 和真实 PhysX 施加相同 SI 执行器目标，比较完整响应，再调整目标端驱动。此阶段沿用用户持续适配授权，不启动新训练；已有 fresh 训练使用冻结 runtime 独立运行。

采用设计：共用不可变输入协议（完整 qpos/qvel、启用关节、重力/接触、分段目标），两端各自推进动力学；按名称映射关节，同时记录 q、步末 qd、区间 Δq/Δt、末态 world/body omega 和原生 MJX gyro。先用悬空无接触单关节辨识，拟合后冻结候选，进行未参与拟合的反向/幅度/组合输入及接地短测。禁止逐步覆写目标状态来伪造匹配，不改质量/惯量/力矩上限或原 JIT。

阶段工程上限：40,000 个 PhysX 物理步（初始化另外显式计入审计）、2,000 个真实 MJX-Warp 物理步；每输入不超过 0.2 s。非有限状态、超预算或接口断言失败即停止保存证据。数据属于适配开发/独立输入工程复核，不是完整任务成功率或独立性能 holdout。

- [ ] `response_protocol.py` + 测试：共享时间线、完整初态、单位和边界检查。
- [ ] `probe_mjx_response.py`：当前训练虚拟环境中的真实 Warp，保留 sensor 与末态角速度。
- [ ] `probe_physx_response.py`：硬重建每例，原增益全隐式/hybrid 的 1/2/5 ms 与当前配置比较；只对有证据的通道调增益。
- [ ] `analyze_response.py` + 测试：仅对齐共同真实采样时刻，联合 q/qd/omega 评分并检查单项退化；保存 CSV、PNG/PDF、误差和全部负结果。
- [ ] 冻结候选、运行新输入复核与接地短测；只有多通道响应实测改善后才决定是否进入策略闭环验证。

产物目录：`results/input_response_20260926`。源基准冻结为 MJX-Warp / RK4 / 5 ms；细步长源对照仅用于诊断，不冒充旧 checkpoint 的原训练动力学。完整 sim2sim 成功目标仍未达到。

同输入首屏实测：MJX 4条×24步=96；PhysX 28条共2112 rollout+56初始化=2168。原增益6种mode/dt组合均未通过q/qd/body-omega联合退化门槛。新假设转为匹配离散执行器响应：在原5ms宏周期中，以实际PhysX状态计算局部标量RK4目标，再用两段真实限幅力矩逼近末态q/v；不读源轨迹回放、不覆写目标状态。源模型静态有效惯量仅计算一次，不改变任何实体惯量。这属于数值执行器适配候选，不是原电机等价。新增同0.5ms/原增益/hybrid对照，第一扩展4候选共2880 rollout步，仍在已声明40k内。所有5ms宏步末及非零初速验证均须保留；验证输入尚未运行。
