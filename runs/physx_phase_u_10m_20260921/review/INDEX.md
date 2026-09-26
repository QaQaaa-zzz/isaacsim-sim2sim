# Best 模型 PhysX 仿真、视频与关节诊断

![全部轨迹](trajectories_all.png)

![每步与累计奖励](rewards_all.png)

训练已完成 **10,000,000** 步。本次固定使用 **transition_05251072**（第 5,251,072 步）；固定开发面板回报 **1355.478**，属于按约25万步间隔参评候选中的最佳。未换用 last，未调参数，未续训。

重新运行原声明的4 ground＋4 airborne测试，32环境布局、完整PhysX重置、确定性Actor、1ms物理/20ms控制。本次与训练结束时保存的轨迹 **0/8 完全逐值一致**，即未通过跨进程逐值重复性检查，详见[重复性](best_rollout/repeatability.json)。平均回报 **1192.156**。全部8回合提前失败；apex只表示达到顶点事件，不能代表越障和落地恢复成功。4个ground为相同物理初态在不同环境位置的重复，不是4种随机工况。

[全部八回合视频总览](all_cases.mp4) · [原训练结果与原Actor对照](../report/INDEX.md) · [本次指标](metrics.json) · [本次实际模型审计](best_rollout/runtime_audit.json)

| seed | 初态 | 终止 s | 原始终止原因 | 回报 | 视频 | 图与数据 |
|---|---|---:|---|---:|---|---|
| 2820701 | ground | 2.24 | roll limit | 1416.13 | [视频](best_rollout/seed_2820701/replay.mp4) | [轨迹](best_rollout/seed_2820701/trajectory.png) / [关节](best_rollout/seed_2820701/joints.png) / [指令](best_rollout/seed_2820701/commands.png) / [力矩](best_rollout/seed_2820701/efforts.png) / [奖励分项](best_rollout/seed_2820701/reward_components.png) / [CSV](best_rollout/seed_2820701/diagnostics.csv) |
| 2820702 | ground | 2.04 | pitch limit | 1620.17 | [视频](best_rollout/seed_2820702/replay.mp4) | [轨迹](best_rollout/seed_2820702/trajectory.png) / [关节](best_rollout/seed_2820702/joints.png) / [指令](best_rollout/seed_2820702/commands.png) / [力矩](best_rollout/seed_2820702/efforts.png) / [奖励分项](best_rollout/seed_2820702/reward_components.png) / [CSV](best_rollout/seed_2820702/diagnostics.csv) |
| 2820703 | ground | 2.42 | roll limit | 1809.47 | [视频](best_rollout/seed_2820703/replay.mp4) | [轨迹](best_rollout/seed_2820703/trajectory.png) / [关节](best_rollout/seed_2820703/joints.png) / [指令](best_rollout/seed_2820703/commands.png) / [力矩](best_rollout/seed_2820703/efforts.png) / [奖励分项](best_rollout/seed_2820703/reward_components.png) / [CSV](best_rollout/seed_2820703/diagnostics.csv) |
| 2820704 | ground | 2.02 | pitch limit | 1573.39 | [视频](best_rollout/seed_2820704/replay.mp4) | [轨迹](best_rollout/seed_2820704/trajectory.png) / [关节](best_rollout/seed_2820704/joints.png) / [指令](best_rollout/seed_2820704/commands.png) / [力矩](best_rollout/seed_2820704/efforts.png) / [奖励分项](best_rollout/seed_2820704/reward_components.png) / [CSV](best_rollout/seed_2820704/diagnostics.csv) |
| 2820723 | airborne | 2.98 | roll limit | 884.74 | [视频](best_rollout/seed_2820723/replay.mp4) | [轨迹](best_rollout/seed_2820723/trajectory.png) / [关节](best_rollout/seed_2820723/joints.png) / [指令](best_rollout/seed_2820723/commands.png) / [力矩](best_rollout/seed_2820723/efforts.png) / [奖励分项](best_rollout/seed_2820723/reward_components.png) / [CSV](best_rollout/seed_2820723/diagnostics.csv) |
| 2820731 | airborne | 3.12 | roll limit | 1111.05 | [视频](best_rollout/seed_2820731/replay.mp4) | [轨迹](best_rollout/seed_2820731/trajectory.png) / [关节](best_rollout/seed_2820731/joints.png) / [指令](best_rollout/seed_2820731/commands.png) / [力矩](best_rollout/seed_2820731/efforts.png) / [奖励分项](best_rollout/seed_2820731/reward_components.png) / [CSV](best_rollout/seed_2820731/diagnostics.csv) |
| 2820733 | airborne | 3.72 | roll limit | 1222.31 | [视频](best_rollout/seed_2820733/replay.mp4) | [轨迹](best_rollout/seed_2820733/trajectory.png) / [关节](best_rollout/seed_2820733/joints.png) / [指令](best_rollout/seed_2820733/commands.png) / [力矩](best_rollout/seed_2820733/efforts.png) / [奖励分项](best_rollout/seed_2820733/reward_components.png) / [CSV](best_rollout/seed_2820733/diagnostics.csv) |
| 2820760 | airborne | 2.44 | yaw limit | -100.00 | [视频](best_rollout/seed_2820760/replay.mp4) | [轨迹](best_rollout/seed_2820760/trajectory.png) / [关节](best_rollout/seed_2820760/joints.png) / [指令](best_rollout/seed_2820760/commands.png) / [力矩](best_rollout/seed_2820760/efforts.png) / [奖励分项](best_rollout/seed_2820760/reward_components.png) / [CSV](best_rollout/seed_2820760/diagnostics.csv) |

每个回合均有PNG/PDF，诊断CSV/NPZ、原始traces.npz、steps.json和endpoint.json。轨迹图是车体根坐标轨迹，不是轮胎接触点轨迹；灰色为此前配对原Actor PhysX轨迹。轨迹不拼接、不延长至8秒，红叉/红线为真实失败终点。指标同时给出原Actor与best共同观察窗口的速度RMSE，避免不同时长直接比较。

关节位置与速度覆盖转向、后轮、被动前轮、髋、膝。四路输出分成归一化策略action与映射后的物理目标：转向/髋/膝为rad，后轮为rad/s。目标在[t,t+20ms)保持，测量值在区间末采集；指令用阶梯图表示。前轮为被动关节，无策略命令。

轮轴位置保留PhysX原始回绕角，因此角度图可能出现跳变；不能据此判定轮子突然反转，应同时查看实测角速度。台阶顶部为0.16m（XML半高度0.16m、中心z=0），图中地面以上高度按模型绘制。

力矩图仅画记录到的髋膝实际显式PD输入（最后1ms物理子步，±30Nm），不是整段20ms峰值。转向和后轮是PhysX原生隐式驱动，当前记录未提供其独立真实马达力矩，不能把目标值、约束反力或估算值冒充实测力矩。功率曲线为奖励使用的力矩×后步速度采样，不是积分机械功。

奖励图保留每步、累计值、全部有符号分项及裁剪/终止回报覆盖调整，因此分项可以核对到总奖励。测试没有新增人为扰动区间；airborne为初态采样。所有视频均是本次真实PhysX闭环轨迹在Isaac中的可视化回放，渲染不计算动力学；失败后冻结并明确标记，不能将冻结时长计作存活。

物理契约沿用训练：轮地各向同性0.5/min（只匹配原初态前向系数）；转向PD2.5/0.4、后轮速度增益0.005、髋膝100/6且±30Nm；模型质量保留。没有因此声称完整各向异性接触一致或sim2sim已经成功。


本次结果解读与检查记录（2026-09-26完成交付，物理轨迹采集于2026-09-22）：

- 地面起步4回合均未稳定通过台阶前缘：根坐标最远X约3.63m，随后后退；2回合侧倾超限、2回合俯仰超限，2.02–2.42s终止。
- 空中初态4回合根坐标最远X为5.73–6.73m，但3回合侧倾超限、1回合偏航超限，2.44–3.72s终止。空中初态有预设上升速度，不能算作策略自主起跳成功。
- 训练末测试平均回报1047.979，本次重测1192.156；两者均为同一best的测试结果，不能将差值当作训练进步。八回合均没有逐值复现：根位置首次差异超过1μm出现在0.46–1.42s，控制目标差异随后出现。历史8环境工程重复检查通过并不能保证当前32环境跨进程重复；本次根因尚未确定，不以调参抹平该差异。详见[逐回合重复性诊断](repeatability_diagnostics.json)。
- 地面回合膝关节最后子步力矩饱和采样占比约11.8%–17.0%；空中四回合该项为0。它支持检查地面起跳与接触阶段的驱动响应，不足以单独证明失败根因。
- 当前最明确的能力缺口是地面起跳后前进/姿态保持，以及空中起始后的落地稳定；奖励排名提升尚未转化为完整任务完成。本次交付没有改变奖励或继续训练。

[交付验证](delivery_verification.json)包含全部8个视频的帧数、25fps、真实失败时刻，以及1049条原JAX奖励重算结果（最大绝对误差约7.13e-6）。[数据验证](data_verification.json)检查动作映射、有限数值、时间步、力矩限幅和累计奖励。每回合video_contact_sheet.png含首帧、1秒和最后一帧，方便检查可视化。摄像机扩大取景以显示抬起的摆臂；仅渲染代码变更，训练物理/策略/奖励哈希保持一致。

复现入口位于ISAAC根目录：evaluate_physx_best.py读取best_model.json并运行原测试面板；plot_physx_review.py生成诊断图；render_traces.py渲染记录轨迹；verify_physx_review.py复算奖励并合成视频。重新仿真需要指定新的输出目录，避免覆盖当前记录。
