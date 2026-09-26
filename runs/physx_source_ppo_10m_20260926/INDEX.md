# 已停止：初始化不符合用户要求

2026-09-26用户明确要求从零重新训练。本运行恢复了MJX预训练参数，属于续训；已停止并保留全部历史记录，不作为从零训练结果。新预算不继承本运行步数或模型。[停止记录](STOPPED_FOR_FRESH_INITIALIZATION.json)。

以下为停止前的运行声明和验证快照：

# 原 MJX/Brax 训练契约迁移至 PhysX：正式运行

2026-09-26启动，训练正在进行。此文是冻结配置与入口，实时计数见[status.json](status.json)。[TensorBoard](http://127.0.0.1:6007/#scalars)中选择 `source_ppo`；`previous`保留上一轮。工程验证通过不表示本轮已经学会越障。

已核实正式第1批：24,576转移、192次Adam，归一化样本数5,013,504；TensorBoard对应标量已读取验证。核实时刻2026-09-26T14:10:59.773562+08:00，不是实时计数。[启动及参数校验](launch_verification.json) · [第一批及TensorBoard验证](first_update_verification.json)。

[本轮原因分析与逐项对照](../../results/ppo_port_audit_20260926/INDEX.md) · [冻结声明](declaration.json) · [best](best_model.json) · [逐批指标](metrics.jsonl) · [实际物理审计](runtime_audit.json) · [实现快照](implementation) · [源库身份](learner_identity.json)

从用户指定的MJX `transition_4988928`恢复Actor、106维特权Critic、完整Welford统计，父模型训练步数4,988,928。新PhysX预算另计9,977,856步，即406个完整块，未把27,648个工程采样转移混入。源checkpoint没有Adam状态，本轮新建原版Adam；这不是从零训练，也不是源优化器精确续训。

| 参数 | 本轮值，与原训练一致 |
|---|---|
| 并行训练环境 | 384，真实CPU PhysX动力学 |
| rollout | 每环境64步，单块24,576转移 |
| minibatch | 每份16条完整64步序列，24份 |
| 每块优化 | 8轮，共192次Adam；无KL提前停止 |
| 网络 | Actor76→256×3→8；Critic106→256×3→1；Swish |
| 学习率 / Adam | 1e-4；b1=.9，b2=.999，eps=1e-8 |
| PPO clip / gamma / lambda | .2 / .99 / .95 |
| entropy / reward scaling / grad cap | .01 / .1 / .5 |
| value MSE有效系数 | .25，复用原Brax损失 |
| 归一化 | 恢复源完整统计，每块采样后、SGD前更新 |
| 初态 | 原JAX随机流，8% airborne RSI；源配置没有额外质量/摩擦随机化 |
| episode / 控制周期 | 400步 / 20ms |

保持上轮已授权的物理：转向PD2.5/0.4、后轮速度增益.005、髋膝显式PD100/6并限±30Nm；各部件质量/惯量保留，轮地静/动摩擦.5与min组合。完整摩擦并不等价：源侧向系数5未保存。PhysX物理步长1ms，源MJX5ms RK4，求解器及离散驱动不同。加速度使用末1ms根速度差分，不是已经证明与MuJoCo传感器等价的数值。训练lane接触缓存完全独立清除尚未证明。

选择规则：每个完整训练块后，独立8环境进程执行固定开发面板（4 ground＋4 airborne）并按平均原始episode回报选best，含初始化，平分保留较早者；每次保存完整learner和可推理Actor。该best面板是用户要求下新增的开发规则，原源训练没有相同的best选择。开发评估不重置384个训练环境。测试另8个冻结种子只在阶段末使用，属于已用过的工程测试面板，不声称独立未知holdout。

重复性限制：工程同策略两次评分6/8轨迹逐值相同，2/8不同；均值343.981与378.727。best只表示本协议实际评过的最高分候选，不是任务合格或小幅差异显著的证明。4个ground种子具有相同物理初态，不能冒称4个独立训练种子。

启动前已完成24项单元/行为测试和13项最终相关回归；768条原JAX奖励逐步复算最大误差4.2916e-6；384环境有限数值采样；完整24,576步+192次Adam工程训练；Actor导出最大误差5.07e-7。工程单块采样142.68s，优化2.14s；实际墙钟受其他项目负载影响，粗估全程约17–20小时，不保证该耗时。

有界流水线会在训练预算完成后固定best及原Actor，保存全部8个测试回合的真实PhysX轨迹、每步奖励及分项、终点和对照图，并自动生成预声明ground/airborne代表视频。训练与可视化阶段分开标记；视频使用本轮PhysX轨迹回放，不使用MJX替代目标端动力学。发生非有限数值、checkpoint导出不符、评估进程失败等情况会记录error并停止，不自动追加预算。

[实时状态](status.json)中completed表示训练和阶段末数值评估完成；最终stage=complete还要求报告及预声明视频验证完成。旧实验、DVGC与STTW代码及环境保持原状；本目录未初始化Git，因此无commit/push。
