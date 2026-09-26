# 已完成：1000万步训练与best仿真交付

训练状态已核验为10,000,000步完成。采用第5,251,072步的best，已重新执行8回合PhysX测试，并生成视频、轨迹、五关节响应、四路动作与物理指令、髋膝实际力矩和奖励分项。

**[完整交付报告](review/INDEX.md) · [八回合视频总览](review/all_cases.mp4) · [全部轨迹](review/trajectories_all.png)**

本次8回合均在2.02–3.72s提前失败；未实现稳定任务完成。与训练末测试未逐值复现，两份记录均保留并在报告说明。以下为原始启动时说明，其中“运行中”和“预定”表述仅代表当时状态。

---

# PhysX 1000 万步训练

从指定 `transition_4988928` 的 Actor/归一化初始化，重新建立 PPO 优化器和价值网络；全部动力学采样在 Isaac Sim 的 PhysX CPU 后端。预算为新增 10,000,000 环境转移，工程试跑不计入。运行中不宣称已训练完成或能力改善。

[实时状态](status.json) · [训练日志](metrics.jsonl) · [当前 best](best_model.json) · [TensorBoard](http://127.0.0.1:6007) · [冻结运行声明](declaration.json) · [实际模型配置](runtime_audit.json)

物理：各部件质量保留；转向 Kp/Kd=2.5/.4、后轮速度增益=.005、髋膝100/6及±30Nm。为保持原能耗奖励的真实力矩输入，髋膝用显式限幅PD，转向/后轮为原生隐式驱动。物理步长1ms，控制20ms。轮地static=dynamic=.5、min组合：匹配初态世界X前行方向；源端另一切向5不保留，原完整各向异性接触并未复现。

复用原奖励、事件及终止函数，8% airborne RSI分布；随机数使用NumPy，不宣称与JAX逐位同序列。训练采用常规并行环境reset；每次计分评估完整重置PhysX，消除接触缓存干扰。14项测试、200条实测奖励重建、8个同Actor完整回合重复性检查已完成，见[工程验证](../../results/isaac_training_engineering/INDEX.md)。

每跨过250,000训练转移，在固定开发8回合上比较原始episode return；包含初始化候选，reward相同保留较早者。4个ground初态相同，4个airborne由固定种子生成；这不是8个独立训练seed或总体成功率估计。最高开发回报不等于任务成功。每50次更新保存latest，开发候选单独保存，最终保存last。

满1000万步后，流水线自动用best和原Actor在另一组8回合配对评估，输出每步奖励、累计奖励、速度、姿态及终点，生成全部条件图和两个预声明代表初态的视频。视频是实际PhysX轨迹回放，终止后冻结并标注；不把MJX运动作为PhysX结果。预定最终报告入口 `report/INDEX.md`，尚未生成前不能认为完成。

[原始代码快照](implementation/) · [实现哈希](implementation_hashes.json)。该目录为独立实验产物，无Git提交/推送，未改DVGC/STTW源码或原checkpoint。
