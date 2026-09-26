"""Rebuild the engineering report from actual completed paired diagnostics."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from select_actuator_calibration import loss
from actuator_reference import velocity_step

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/actuator_report'
OUT.mkdir(exist_ok=True)
AUDIT = ROOT/'results/actuator_audit'
CAL = ROOT/'results/actuator_calibration'
VAL = ROOT/'results/actuator_calibration_validation'
POL = ROOT/'results/actuator_policy_smoke'
audit = json.loads((AUDIT/'summary.json').read_text())
selected = json.loads((CAL/'selection.json').read_text())
policy = json.loads((POL/'summary.json').read_text())
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .25})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT/(name+'.png'), dpi=170)
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)


def traces(folder):
    return np.load(folder/'traces.npz')


# The state channels are deliberately separate: q alone is misleading here.
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for col, dt in enumerate([.005, .0001]):
    name = f'rotor_servo_{dt:g}'
    d = traces(AUDIT/name)
    native = np.load(ROOT/'results/actuator_mjx_rotor'/name/'trace.npz')['mjx']
    for row, component in enumerate([1, 2]):
        ax = axes[row, col]
        dense_t = np.linspace(0., .02, 2001)
        analytic_q, analytic_v = velocity_step(dense_t, inertia=.0004165, damping=5., limit=6., target=12.)
        ax.plot(dense_t*1000, [analytic_q, analytic_v][row], 'k-', label='Continuous analytic')
        for engine, label, style in [('RK4', 'MuJoCo CPU RK4', 'C0-'), ('physx', 'PhysX', 'C1--')]:
            ax.plot(d[engine][:, 0]*1000, d[engine][:, component], style, label=label)
        ax.plot(native[:, 0]*1000, native[:, component], 'o', color='C2', markersize=3,
                markevery=max(1, len(native)//10), label='Native MJX-Warp')
        ax.set(xlabel='Time (ms)', ylabel=['Angle (rad)', 'Velocity (rad/s)'][row], title=f'Original isolated rear rotor | dt={dt*1000:g} ms')
        if row == 0 and col == 0: ax.legend(fontsize=8)
save(fig, 'rotor_state_channels')

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
names = ['rearwheel_joint', 'steering_joint', 'hip_joint', 'knee_joint']
for ax, joint in zip(axes.flat, names):
    rs = sorted([r for r in audit if r['case']['model']=='bike' and r['case']['mode']=='servo' and r['case']['joint']==joint], key=lambda r:r['case']['dt'])
    dt = [r['case']['dt']*1000 for r in rs]
    for integrator, label in [('RK4','PhysX vs source RK4'),('IMPLICITFAST','PhysX vs source implicitfast')]:
        ax.loglog(dt, [r[integrator]['qd_rmse_rad_s'][r['joint_names'].index(joint)] for r in rs], 'o-', label=label)
    ax.set(xlabel='Physics timestep (ms)', ylabel='Joint velocity RMSE (rad/s)', title=joint)
    ax.legend(fontsize=8)
save(fig, 'servo_timestep_convergence')

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for col, joint in enumerate(names[:2]):
    entry = selected[joint]
    for role, label in [('baseline','PhysX original gains'),('best','PhysX fitted gains')]:
        d = traces(CAL/entry[role]['case']['name']); ns = d['joint_names'].tolist(); i = ns.index(joint); n = len(ns)
        for row, component in enumerate([1+i, 1+n+i]):
            axes[row, col].plot(d['physx'][:, 0]*1000, d['physx'][:, component], label=label)
            if role == 'baseline':
                axes[row, col].plot(d['RK4'][:, 0]*1000, d['RK4'][:, component], 'k--', label='Frozen source RK4 (MJX checked)')
    for row in range(2):
        axes[row, col].set(title=joint, xlabel='Time (ms)', ylabel=['Angle (rad)','Velocity (rad/s)'][row])
        axes[row, col].legend(fontsize=8)
save(fig, 'calibration_tradeoff')

validation_rows = []
for i, case in enumerate(json.loads((CAL/'declaration.json').read_text())['validation']):
    before = loss(VAL/f'validation_{i}_baseline'); after = loss(VAL/f'validation_{i}_best')
    validation_rows.append(dict(index=i, **case, **{'baseline_'+k:v for k,v in before.items()}, **{'fitted_'+k:v for k,v in after.items()}))
with (OUT/'validation.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(validation_rows[0])); w.writeheader(); w.writerows(validation_rows)
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, r in zip(axes.flat, validation_rows):
    i = r['index']
    for role, label in [('baseline','Original gains'),('best','Fitted gains')]:
        d = traces(VAL/f'validation_{i}_{role}'); ns=d['joint_names'].tolist(); j=ns.index(r['joint']); n=len(ns)
        ax.plot(d['physx'][:, 0]*1000, d['physx'][:, 1+n+j], label=label)
        if role=='baseline': ax.plot(d['RK4'][:, 0]*1000, d['RK4'][:, 1+n+j], 'k--', label='Frozen source RK4')
    ax.set(title=f"{r['joint']}\ninput={r['input']:g}, initial qd={r['initial_velocity']:g}", xlabel='Time (ms)', ylabel='Velocity (rad/s)')
    ax.legend(fontsize=8)
save(fig, 'independent_inputs_velocity')

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for r in policy:
    tr = traces(POL/r['case'])['target']
    w,x,y,z = tr[:,4:8].T
    roll = np.degrees(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)))
    axes[0,0].plot(tr[:,1],tr[:,2],label=r['case']); axes[0,0].plot(tr[-1,1],tr[-1,2],'x')
    for ax, values in [(axes[0,1],tr[:,18]),(axes[1,0],roll),(axes[1,1],tr[:,3])]:
        ax.plot(tr[:,0],values,label=r['case']); ax.plot(tr[-1,0],values[-1],'x')
axes[0,0].set(xlabel='World X (m)',ylabel='World Y (m)',title='Original policy: XY until actual failure')
axes[0,1].set(xlabel='Time (s)',ylabel='Root origin vx (m/s)',title='Forward speed'); axes[0,1].axhline(2,color='k',ls=':',label='2 m/s reference')
axes[1,0].set(xlabel='Time (s)',ylabel='Roll (deg)',title='Actual roll-limit failures'); axes[1,0].axhline(35,color='r',ls=':'); axes[1,0].axhline(-35,color='r',ls=':')
axes[1,1].set(xlabel='Time (s)',ylabel='Root height (m)',title='Neither candidate reaches task apex')
for ax in axes.flat: ax.legend(fontsize=8)
save(fig, 'original_policy_smoke')

passive = [r for r in audit if r['case']['model']=='bike' and r['case']['mode']=='torque']
passive_error = max(r['RK4']['qd_relative_l2'] for r in passive)
rotor_torque = [r for r in audit if r['case']['model']=='rotor' and r['case']['mode']=='torque']
rotor_error = max(r['physx_analytic']['velocity_relative_l2'] for r in rotor_torque)
native_results = sum([json.loads((ROOT/'results'/folder/'summary.json').read_text()) for folder in ['actuator_mjx_rotor','actuator_mjx_bike']], [])
verified = dict(no_training=True, actuator_probe_cases=len(audit), calibration_fit_cases=34,
                independent_validation_cases=12, original_policy_smoke_cases=len(policy), native_mjx_cases=len(native_results),
                passive_velocity_relative_l2_max=passive_error, passive_gate_passed=passive_error<=.01,
                fixed_rotor_relative_velocity_error_max=rotor_error, fixed_rotor_gate_passed=rotor_error<=.001,
                native_mjx_cpu_max_abs_qd_difference=max(r['max_abs_qd_difference_from_cpu'] for r in native_results),
                fitted_candidate_adopted=False, sim2sim_accepted=False,
                matched_scope='contact-free short probes only; no proof of contact-model equivalence',
                policy_results=policy, validation=validation_rows)
(OUT/'summary.json').write_text(json.dumps(verified, indent=2))
rows = []
for r in audit:
    rows.append(dict(name=r['case']['name'], model=r['case']['model'], mode=r['case']['mode'], joint=r['case']['joint'], dt=r['case']['dt'], duration=r['case']['duration'], relative_qd_error_RK4=r['RK4']['qd_relative_l2']))
with (OUT/'audit_cases.csv').open('w') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fit_table = '\n'.join(f"| {j} | {r['best']['case']['target_drive_scales']} | {r['baseline']['q_rmse_rad']:.6f} → {r['best']['q_rmse_rad']:.6f} | {r['baseline']['qd_rmse_rad_s']:.4f} → {r['best']['qd_rmse_rad_s']:.4f} |" for j,r in selected.items())
val_table = '\n'.join(f"| {r['index']} | {r['joint']} | {r['input']:g} / {r['initial_velocity']:g} | {r['baseline_loss']:.4f} → {r['fitted_loss']:.4f} | {r['baseline_q_rmse_rad']:.6f} → {r['fitted_q_rmse_rad']:.6f} | {r['baseline_qd_rmse_rad_s']:.4f} → {r['fitted_qd_rmse_rad_s']:.4f} |" for r in validation_rows)
policy_table = '\n'.join(f"| {r['case']} | {r['duration']:.2f} | {r['target_endpoint']['end_code']} | {r['target_endpoint']['jump_zone_seen']} | {r['target_endpoint']['apex_seen']} |" for r in policy)
text = f'''# 执行器响应诊断与 Isaac 参数标定 — 2026-09-20

**本阶段找到明确的离散执行器响应差异，并完成一轮目标参数标定；sim2sim 仍未通过。** 原 checkpoint 未更换，本轮训练交互为 0。

![原策略标定前后实际轨迹](original_policy_smoke.png)

[标定前后视频](comparison.mp4) · [角度与速度诊断](rotor_state_channels.png) · [全部结果 JSON](summary.json) · [45 组诊断 CSV](audit_cases.csv) · [独立输入验证 CSV](validation.csv)

## 1. 哪些已验证，哪些没有

- 20 组整车悬空恒力矩试验：5 个关节 × 正负 0.02 Nm × 1/0.1 ms；各 50 ms。禁用重力、接触和所有伺服，零初速，hip=-0.8、knee=1.0 以避开限位。最大关节速度相对 L2 差异 **{passive_error*100:.6f}%**，通过预声明 1% 工程门槛。它支持当前质量/惯量/关节映射在这些自由运动工况下数值一致，不证明全状态或接触一致。
- 4 组固定轮轴恒力矩试验：目标速度相对解析解最大误差 **{rotor_error*100:.6f}%**，通过 0.1% 工程门槛。固定支撑是单独生成的诊断夹具；轮轴质量和惯量来自原模型。
- 16 组整车单执行器试验：4 个主动关节 × 5/1/0.1/0.05 ms，非测试伺服关闭，重力/接触关闭。
- 5 组孤立轮轴伺服试验：5/1/0.5/0.1/0.05 ms。所有步长均单独记录 q、qd，不能用视觉或单独 q 判断等价。
- 4 组在原训练环境运行的原生 MJX-Warp 核对：孤立轮轴 5/0.1 ms、整车后轮/转向 5 ms。和 CPU RK4 的关节速度最大绝对差 **{verified['native_mjx_cpu_max_abs_qd_difference']:.6g} rad/s**。其余源端测试仍是 CPU RK4，不冒称全部由 MJX 执行。

## 2. 关键发现：5 ms 时角度运动与步末速度出现严重偏离

![同一轮轴角度和速度](rotor_state_channels.png)

孤立原后轮 I=0.0004165 kg·m²，速度反馈增益 5 Nm/(rad/s)，力矩上限 ±6 Nm，目标 12 rad/s。20 ms 后：

- 源端 RK4 5 ms：q≈0.240096 rad，qd≈0；原生 MJX-Warp 复现这一现象。
- PhysX 5 ms：q≈0.239000 rad，qd≈12 rad/s。
- 源端 RK4 0.1 ms：qd≈12 rad/s；0.05 ms 对连续解析解速度 RMSE≈0.000084 rad/s。

**不能说源端轮子不转；它的角度确实在转，但返回的步末速度接近零。** 这也解释了为什么画面相似可能掩盖策略输入差异。5 ms 比该孤立伺服线性时间常数 I/k≈0.0833 ms 大约 60 倍；饱和力矩在 RK4 内部阶段切换。这里有解析解、步长细化和原生 MJX 复现共同支撑，不是凭视频推断。

物理方程解析参考为 I·dv/dt=clip(k·(v_target-v), ±6)，q 为 v 的连续积分。先以恒定力矩加速，进入未饱和区后指数趋近目标。目标端孤立轮轴的 I·Δqd/dt 只表示**一步平均净力矩**；源端保存的 qfrc_actuator 是刷新后的瞬时值，两者不能混称同一时刻电机力矩。整车 PhysX 电机力矩未单独测出。

![四个伺服步长细化](servo_timestep_convergence.png)

整车悬空后轮跨引擎 qd RMSE：5 ms 为 11.4403 rad/s，0.1 ms 为 0.000898 rad/s；转向为 3.9886 → 0.002235 rad/s。这里含初始采样，标定表的 RMSE 排除 t=0，因此数值略不同。不同 dt 的采样网格不同，解析图中粗步长会漏掉快速瞬态；不能据某个步末速度误差小断言粗步长准确。最小步长也存在 PhysX 浮点误差平台，不保证每个关节严格单调收敛。

PhysX 官方说明其关节驱动使用隐式约束；MuJoCo 对 RK4 与 implicitfast 的适用性有明确区分：[PhysX Articulations](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html)、[MuJoCo 3.3.7 Computation](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html)。更改源端步长会改变原 checkpoint 所处的环境；本轮仅诊断，未静默替换训练物理。

## 3. 按用户授权调 Isaac 参数逼近冻结源端

目标端单独扫描：后轮 kd 倍率 0、1e-6、1e-5、1e-4、0.001、0.01、0.1、0.5、1（9 组）；转向 kp/kd 各取 0.25、0.5、1、2、4（25 组）。总 34 组，每组 50 ms。原质量、惯量、力矩上限不变；没有给策略回放源端运动，没有接管 PhysX 状态。

目标函数提前声明为角度和速度的归一化平方误差之和；后轮尺度 0.6 rad、12 rad/s，转向尺度 0.1 rad、2 rad/s，时间点排除 t=0。它是工程拟合指标，不是任务奖励或安全门槛。源端模型和输入不随目标参数改变。

| 关节 | 所选目标增益倍率 | q RMSE（rad，原→拟合） | qd RMSE（rad/s，原→拟合） |
|---|---|---|---|
{fit_table}

实际参数：后轮 kv **5→0.005**；转向 kp **10→2.5**、kd **0.1→0.4**。后轮增益降低了 1000 倍，只能视为数值标定候选，不能解释为真实车辆电机参数。转向最优点位于本次搜索边界，后轮搜索是离散网格；均不代表全局最优。

![标定得到的角度与速度取舍](calibration_tradeoff.png)

加权误差降低，但代价是多个工况的角度误差明显增大。现有结果只表明**该组静态增益没有同时对齐角度和速度**，不证明所有可能的 Isaac 参数或执行器模型都无法拟合。

## 4. 未参与拟合的六组输入

选择完成后冻结参数，对预先声明的其他目标值和初速度运行原参数/候选配对，各 50 ms，共 12 个目标回合。没有根据这里的结果再选参数。它们是独立诊断输入，不是大样本泛化测试。

| 编号 | 关节 | 目标输入 / 初始速度 | 归一化 loss（原→拟合） | q RMSE（rad，原→拟合） | qd RMSE（rad/s，原→拟合） |
|---|---|---|---|---|---|
{val_table}

后轮输入单位 rad/s；转向输入为相对初始角度的 rad；初始速度均 rad/s。完整逐步 q/qd 保存在 [原始验证目录](../actuator_calibration_validation/)。

![独立输入逐步速度](independent_inputs_velocity.png)

## 5. 加载指定原策略的真实 PhysX 闭环检查

checkpoint：`phase_u_v4_speed2_roll400_missed200_9977856_seed820701_20260826/checkpoints/transition_4988928`。观测来自 PhysX，使用原 Actor 与归一化，保留原动作映射及 20 ms 控制周期；仅候选的目标驱动增益改变。接触仍是当前 PhysX 标量摩擦模型，未完成源端各向异性接触标定。

| 配置 | 实际终止时间（s） | end_code | 进入跳跃区 | 原任务 apex |
|---|---|---|---|---|
{policy_table}

两者均以侧倾超限（code=3）失败。0.32→0.96 s 表示本次名义短测试的失败延后，**不表示成功率提升，也不能把存活时间三倍解释为迁移能力三倍**。未完成起跳，因此不采用候选作为已通过配置。

[视频](comparison.mp4) 是上述**真实 PhysX 轨迹的渲染回放**，渲染时不执行物理；左边原参数、右边标定参数。它不是 MJX 轨迹冒充 PhysX；真实失败后保持最后姿态，并明确显示终止时间。图中的数据在真实失败处截止。

## 6. 后续执行顺序

1. 继续目标端标定，但使用 q、qd、控制响应同时约束，增加混合执行器输入和外部负载；不能靠修改观测符号、回放源端姿态或只拟合视频获得假通过。当前静态增益结果保留为负结果和部分改进证据。
2. 单独标定轮地接触：先做滑移/制动/滚动和法向加载曲线，再识别目标摩擦、接触偏置及可表达的耗散参数；不要同时用碰撞误差补偿伺服误差。源端 [5,0.5] 两切向摩擦与目标标量5不是同一模型。
3. 如果被冻结的粗步长源端 q/qd 无法由物理合理的目标执行器同时逼近，应建立双方可复现的离散执行器/观测契约，并把旧策略适配放在新实验分支中明确验证。不能静默改掉原训练物理或把该问题仅归因于缺少域随机化。
4. 在辨识残差范围内设计域随机化，再做独立面板的任务跟踪与跳跃评估。本轮无训练、无新成功声明。

## 复现与实现

- [诊断配置](../../configs/actuator_audit.json)、[标定声明](../../configs/actuator_calibration.json)、[冻结选择](../actuator_calibration/selection.json)、[策略测试配置](../../configs/actuator_policy_smoke.json)。
- [诊断入口](../../actuator_audit.py)、[原生 MJX 核对](../../actuator_mjx_check.py)、[解析参考](../../actuator_reference.py)、[命名关节增益映射](../../drive_calibration.py)、[报告生成器](../../report_actuator_audit.py)。
- 使用独立 `.venv` 执行 Isaac；原生 MJX 核对使用原训练环境的已有依赖，未安装/改动该环境。模型/检查点哈希另见 `verification.json`。
- 输出目录拒绝覆盖已有诊断；复现时选新 `--output` 路径。全部源码和产物在 `/home/qy/ISAAC——SIM`，当前根目录不是 Git 仓库，没有 commit/push。
'''
(OUT/'INDEX.md').write_text(text)
print(json.dumps({k:v for k,v in verified.items() if k not in ['policy_results','validation']}, indent=2))
