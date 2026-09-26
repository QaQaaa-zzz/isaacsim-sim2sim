# 中断后恢复本轮从零训练

恢复来源：本轮`checkpoints/transition_00073728`，3个完整PPO更新，576次Adam，归一化统计73,728样本。恢复网络、完整优化器、归一化及策略采样随机状态；初始训练身份仍为fresh，未加载MJX预训练模型。

[实时状态](../../status.json) · [恢复前状态](status_before_resume.json) · [恢复声明](resume.json) · [learner恢复核验](learner_restore_verification.json) · [预检与来源](preflight.json) · [日志](training.log)

原总预算9,977,856转移，恢复后剩余9,904,128转移，从第4批继续。原best为第24,576步，继续保留并参与后续开发排名。原declaration、checkpoint、指标、评估与旧代码快照不覆盖；后续指标/新checkpoint接在同一实验中。

中断前状态最后记录74,112采样转移，其中至少384个属于未提交采样；实际丢弃量可能更多，进程未记录准确退出点。这些交互不能伪装成从未发生，也不能当已完成PPO更新。恢复计数以完整保存的73,728转移为准，总预算不增加。

checkpoint没有PhysX完整求解器/接触缓存或回合现场，恢复会重建全部训练场景和观测历史。reset随机流推进到192个已提交控制tick的位置，策略/优化器随机状态从checkpoint恢复；不声称物理轨迹逐位连续。首次恢复后的训练分布受重新初始化回合影响。

已确认训练、评估、TensorBoard进程均退出，日志无Python异常，未找到对应的内核OOM/GPU错误；退出原因尚未确定。此前仅用setsid脱离终端，并未创建独立服务；当前执行终端的cgroup属于界面进程。这提示需要独立服务管理，但不足以证明本次退出由界面关闭导致。此次运行使用独立systemd用户服务`isaac-physx-fresh-20260926-r01.service`；不配置无限自动重试。TensorBoard由`isaac-tensorboard-6007-r01.service`管理。

执行目录runtime冻结原训练的物理、观测、随机流、learner和资产，仅训练入口与pipeline补充恢复支持。这样不会把另行开展的迁移诊断修改混入已声明训练。13项恢复/learner测试通过，包括拒绝损坏checkpoint、拒绝配置变化和旧checkpoint回退、优化器/随机流保存恢复。

工程参数和原奖励不变；本次恢复不构成任务能力提升证据。当前仍需继续训练并在阶段末完成best测试与可视化。

恢复运行已验证（2026-09-26T14:48:42.125289+08:00）：第4批完成，累计98,304训练转移、768次Adam，统计样本数98,304；新checkpoint哈希与Actor导出通过，独立开发评估完成，TensorBoard已加载对应新标量。独立用户服务仍在运行。[验证记录](verification.json)。
