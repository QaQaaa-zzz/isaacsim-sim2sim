# Isaac PhysX 原奖励训练与 best 评估

已完成训练转移 10,000,000 / 10,000,000。真实动力学为 PhysX；best 是固定开发面板已评估候选（含原始初始化）中奖励最高者，不代表任务合格。

best: `/home/qy/ISAAC——SIM/runs/physx_phase_u_10m_20260921/checkpoints/transition_05251072`，开发平均原始回报 1355.478。实际训练配置 [declaration.json](../declaration.json)。原 Actor 初始化，新优化器/价值网络，非原 Brax 优化器精确续训。

轮地各向同性摩擦 0.5（仅匹配原初态前进方向）；转向2.5/0.4、后轮速度反馈0.005；髋膝100/6、±30Nm显式PD，实际输入力矩用于原能耗项。物理1ms、控制20ms。质量不变，其他接触模型差异未消失。

![训练曲线](training.png)

每个面板包括XY、速度、高度、侧倾、每步奖励和累计奖励。图在真实终止处截止；4个ground种子产生相同确定性初态，不能视为4个独立训练种子。4个airborne初态由冻结种子决定；开发和测试seed分开，测试不用于选择best。

[seed 2820701](seed_2820701.png) · [seed 2820702](seed_2820702.png) · [seed 2820703](seed_2820703.png) · [seed 2820704](seed_2820704.png) · [seed 2820723](seed_2820723.png) · [seed 2820731](seed_2820731.png) · [seed 2820733](seed_2820733.png) · [seed 2820760](seed_2820760.png)

| seed | policy | return | duration s | speed RMSE m/s | end_code | apex |
|---|---|---:|---:|---:|---|---|
| 2820701 | original | 130.03 | 1.76 | 1.590 | 3 | True |
| 2820701 | best | 1389.07 | 2.22 | 1.963 | 3 | True |
| 2820702 | original | 513.86 | 1.50 | 0.872 | 3 | True |
| 2820702 | best | 1632.14 | 2.08 | 2.134 | 4 | True |
| 2820703 | original | -100.00 | 0.74 | 0.570 | 8 | False |
| 2820703 | best | 1537.71 | 2.40 | 2.047 | 3 | True |
| 2820704 | original | 489.61 | 1.48 | 0.860 | 3 | True |
| 2820704 | best | 1665.08 | 2.10 | 2.017 | 4 | True |
| 2820723 | original | -100.00 | 1.38 | 0.468 | 10 | True |
| 2820723 | best | -100.00 | 2.86 | 0.927 | 10 | True |
| 2820731 | original | 740.06 | 1.06 | 0.423 | 3 | True |
| 2820731 | best | 1207.88 | 3.58 | 1.093 | 3 | True |
| 2820733 | original | 504.98 | 1.04 | 0.304 | 3 | True |
| 2820733 | best | 1151.95 | 3.54 | 1.061 | 3 | True |
| 2820760 | original | -100.00 | 1.26 | 0.515 | 10 | True |
| 2820760 | best | -100.00 | 2.40 | 0.964 | 10 | True |

原始逐步数据及分项CSV在 `evaluation/original_test` 和 `evaluation/best_test`。视频是这些真实PhysX轨迹的标注回放，渲染不执行物理，终止后冻结姿态。仅展示首个ground/airborne种子的预声明代表视频；全部8个条件都保留图、数据和终点。
