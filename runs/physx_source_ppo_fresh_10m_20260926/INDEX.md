# PhysX 从零重新训练

**当前：中断后已从本实验73,728步checkpoint恢复。** 接续第4批，原总预算不变；恢复模型、优化器与统计，物理场景重建。[恢复记录](recovery/attempt_0001/INDEX.md)。

本轮按用户纠正执行：**Actor、Critic全部随机初始化，没有加载MJX或任何Isaac checkpoint的权重**。归一化初始count=0、mean=0、std=1；Adam动量与计数为0。原`transition_4988928`只用于定位奖励和训练配置，不是初始化模型。

正式第一批已核实：24,576转移、192次Adam，统计计数=24,576，时间2026-09-26T14:30:56.727595+08:00。[实际零步参数验证](fresh_initialization_verification.json) · [第一批及TensorBoard验证](first_update_verification.json)。

[实时状态](status.json) · [best](best_model.json) · [初始化身份](learner_identity.json) · [零步checkpoint身份](checkpoints/transition_00000000/identity.json) · [冻结配置](declaration.json) · [TensorBoard](http://127.0.0.1:6007/#scalars)

TensorBoard选择`fresh`。`cancelled_transfer`是已停止的错误续训，`previous`是更早的已完成实验；不能混合三者步数或模型。

本轮独立预算9,977,856个训练转移（406个完整块，约1000万）。384并行环境×64步；每块24个minibatch、每份16条完整序列，优化8轮，共192次Adam。学习率1e-4、clip.2、gamma.99、lambda.95、熵系数.01、奖励缩放.1、梯度上限.5。Actor76→256×3→8，Critic106→256×3→1，Swish，使用原Brax初始化、损失、归一化和优化机制。种子820701。

原奖励、400步回合、8% airborne RSI及已批准物理配置保持。控制20ms、PhysX物理1ms；转向PD2.5/.4、后轮速度增益.005、髋膝PD100/6并限±30Nm，质量不变，轮地摩擦.5/min。侧向摩擦5、MJX积分器及加速度传感器数值并未完整等价，训练结果仍需真实PhysX评估。

每个完整块在独立8环境进程评分并选best（含随机初始化），评估不重置训练环境。阶段末固定best做8个测试回合及随机初始化对照，生成轨迹、奖励图与预声明ground/airborne代表视频；目录`original_test`在本轮表示随机初始化策略。best只是开发回报排名，不能直接称任务成功。

启动前26项测试通过，包含原Brax随机初始化一致性、统计从零累计、无预训练父模型、保存恢复，以及原PPO/奖励回归。完整工程块24,576转移、192次Adam完成，normalizer_count=24,576；正式启动再次随机初始化，不使用工程块训练后的参数。[初始化验证](../../results/ppo_port_audit_20260926/fresh_initialization_verification.json) · [工程与启动门槛](../../results/ppo_port_audit_20260926/fresh_launch_gate.json)。

工程首块解析KL均值492.796，较大；这里只证明初始化、有限数值、更新和导出流程通过，不证明优化稳定或任务已学会。保留原配置无KL提前停止机制，指标持续记录在metrics.jsonl/TensorBoard。工程阶段以及先前错误续训不计入本轮正式步数。

[已停止续训记录](../physx_source_ppo_10m_20260926/STOPPED_FOR_FRESH_INITIALIZATION.json)。该运行最后记录132,480采样转移、5个完整更新；停止时可能另有未写入状态的少量采样，未伪造精确停止计数。旧数据保留，仅作审计。
