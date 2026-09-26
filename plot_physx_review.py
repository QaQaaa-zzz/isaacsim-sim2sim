"""Detailed finite-episode diagnostics from actual PhysX best-policy logs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from policy_runtime import control
import mujoco

ROOT = Path(__file__).resolve().parent
REASONS = {3: 'roll limit', 4: 'pitch limit', 7: 'backward exit', 8: 'stuck', 9: 'timeout', 10: 'yaw limit', 11: 'jump zone missed'}
JOINTS = ['steering_joint', 'rearwheel_joint', 'frontwheel_joint', 'hip_joint', 'knee_joint']
ACTUATORS = ['steering_joint', 'rearwheel_joint', 'hip_joint', 'knee_joint']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    out = args.input.parent
    best = json.loads((args.run/'best_model.json').read_text())
    checkpoint = Path(best['checkpoint']).name
    cfg = json.loads((ROOT/'policy/resolved_config.json').read_text())
    m = mujoco.MjModel.from_xml_path(str(ROOT/'model/source.xml'))
    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25})
    summary = json.loads((args.input/'summary.json').read_text())
    fig_xy, ax_xy = plt.subplots(2, 2, figsize=(13, 9))
    fig_rewards, ax_rewards = plt.subplots(2, 2, figsize=(13, 8))
    metrics, checks, links = [], [], []
    for endpoint in summary:
        seed = endpoint['seed']; name = f'seed_{seed}'; folder = args.input/name
        zfile = np.load(folder/'traces.npz'); z = zfile['target']; names = list(zfile['joint_names'])
        rows = json.loads((folder/'steps.json').read_text())
        oldfolder = args.run/'evaluation'/'original_test'/name
        old = np.load(oldfolder/'traces.npz')['target']
        oldrows = json.loads((oldfolder/'steps.json').read_text())
        t = z[:, 0]; steps = np.arange(1, len(z)+1); dt = .02
        action = np.array([r['action'] for r in rows]); ctrl = z[:, 27:31]
        state = np.array([r['reward_state'] for r in rows]); effort = state[:, -2:]
        reward = np.array([r['reward'] for r in rows]); oldreward = np.array([r['reward'] for r in oldrows])
        components = np.array([list(r['components'].values()) for r in rows])
        keys = list(rows[0]['components'])
        rawsum = components.sum(axis=1)
        adjustment = reward-rawsum  # clipping and terminal return overrides are explicit
        assert len(z)==len(rows)==endpoint['steps']
        assert np.isfinite(z).all() and np.isfinite(action).all() and np.isfinite(effort).all()
        np.testing.assert_allclose(t, steps*dt, atol=1e-12)
        np.testing.assert_allclose(ctrl, np.array([control(a, m, cfg) for a in action]), atol=2e-6)
        np.testing.assert_allclose(reward.sum(), endpoint['return'], atol=1e-8)
        np.testing.assert_allclose(state[:, -4:-2], z[:, 13+np.array([names.index('hip_joint'),names.index('knee_joint')])], atol=1e-6)
        assert np.abs(effort).max()<=30.00001 and np.abs(action).max()<=1.000001
        assert rows[-1]['terminated'] or rows[-1]['truncated']
        reason = REASONS.get(endpoint['end_code'], str(endpoint['end_code']))
        reset = 'airborne' if endpoint['airborne_reset'] else 'ground'
        label = f'{seed} ({reset})'
        title = f'BEST {checkpoint} | seed {seed} | {reset} | {reason} at {t[-1]:.2f}s'
        def save(fig, filename, time_axes=None):
            if time_axes is not None:
                for axis in np.asarray(time_axes).ravel():
                    axis.axvline(t[-1], color='crimson', linestyle=':', linewidth=1)
                    axis.set_xlabel('Time (s)'); axis.set_xlim(0,t[-1]+.02)
            fig.suptitle(title, fontsize=11)
            fig.tight_layout(rect=[0,0,1,.96])
            fig.savefig(folder/(filename+'.png'), dpi=150)
            fig.savefig(folder/(filename+'.pdf'))
            plt.close(fig)
        fig, axes = plt.subplots(2,3,figsize=(15,8))
        for axis, ix, iy, xlabel, ylabel in [(axes[0,0],1,2,'X (m)','Y (m)'), (axes[0,1],1,3,'X (m)','Z (m)')]:
            axis.plot(old[:,ix],old[:,iy],color='.6',label='Original actor')
            axis.plot(z[:,ix],z[:,iy],label='Best actor')
            axis.scatter(z[-1,ix],z[-1,iy],marker='x',color='crimson',label='Actual failure')
            axis.set(xlabel=xlabel,ylabel=ylabel);axis.legend()
        axes[0,0].set_aspect('equal',adjustable='datalim')
        obstacle=m.geom('step');left=obstacle.pos[0]-obstacle.size[0];top=obstacle.pos[2]+obstacle.size[2]
        axes[0,1].add_patch(Rectangle((left,0),2*obstacle.size[0],top,color='.5',alpha=.25))
        axes[0,1].axhline(0,color='.4');axes[0,1].set_xlim(min(z[:,1].min(),old[:,1].min())-.1,max(4,z[:,1].max(),old[:,1].max())+.1)
        axes[0,2].plot(t,z[:,18],label='Root world-X velocity');axes[0,2].axhline(2,color='k',ls='--',label='Reward target 2 m/s');axes[0,2].set_ylabel('Velocity (m/s)');axes[0,2].legend()
        for j, angle in enumerate(['roll','pitch','yaw']): axes[1,0].plot(t,np.rad2deg(state[:,3+j]),label=angle)
        axes[1,0].set_ylabel('Orientation (deg)');axes[1,0].legend()
        axes[1,1].plot(steps,reward,label='Best');axes[1,1].plot(np.arange(1,len(oldreward)+1),oldreward,color='.6',label='Original');axes[1,1].set(xlabel='Control steps',ylabel='Actual per-step reward');axes[1,1].legend()
        axes[1,2].plot(steps,np.cumsum(reward));axes[1,2].plot(np.arange(1,len(oldreward)+1),np.cumsum(oldreward),color='.6');axes[1,2].set(xlabel='Control steps',ylabel='Cumulative reward')
        for a in [axes[0,2],axes[1,0]]: a.axvline(t[-1],color='crimson',ls=':');a.set_xlabel('Time (s)')
        save(fig,'trajectory')
        fig, axes = plt.subplots(5,2,figsize=(14,15))
        for j,joint in enumerate(JOINTS):
            index=names.index(joint)
            axes[j,0].plot(t,z[:,8+index],label='Measured position (wrapped)' if 'wheel' in joint else 'Measured position');axes[j,0].set_ylabel(joint+' (rad)')
            axes[j,1].plot(t,z[:,13+index],label='Measured velocity');axes[j,1].set_ylabel(joint+' (rad/s)')
            if joint in ACTUATORS:
                k=ACTUATORS.index(joint);axis=axes[j,1 if k==1 else 0]
                axis.stairs(ctrl[:,k],np.r_[0,t],baseline=None,color='darkorange',label='Command over control interval',linewidth=1)
            else: axes[j,0].set_title('Passive front wheel: no motor target')
            for axis in axes[j]:axis.legend(fontsize=7)
        save(fig,'joints',axes)
        fig, axes = plt.subplots(4,2,figsize=(14,11))
        for j,joint in enumerate(ACTUATORS):
            axes[j,0].stairs(action[:,j],np.r_[0,t],baseline=None,label=joint);axes[j,0].set_ylim(-1.05,1.05);axes[j,0].set_ylabel('Normalized action');axes[j,0].legend()
            axes[j,1].stairs(ctrl[:,j],np.r_[0,t],baseline=None,color='darkorange');axes[j,1].set_ylabel('Target '+('rad/s' if j==1 else 'rad'));axes[j,1].set_title(joint)
        save(fig,'commands',axes)
        fig, axes = plt.subplots(2,2,figsize=(13,7))
        for j,joint in enumerate(['hip_joint','knee_joint']):
            axes[j,0].plot(t,effort[:,j]);axes[j,0].axhline(30,color='.5',ls='--');axes[j,0].axhline(-30,color='.5',ls='--');axes[j,0].set_ylabel(joint+' applied torque (Nm)')
            axes[j,1].plot(t,effort[:,j]*state[:,-4+j]);axes[j,1].set_ylabel(joint+' torque x post-step speed (W)')
        axes[0,0].set_title('Actual motor input at last 1ms substep')
        axes[0,1].set_title('Reward power sample; not interval-integrated work')
        save(fig,'efforts',axes)
        fig, axes=plt.subplots(3,2,figsize=(15,11))
        for j, indices in enumerate(np.array_split(np.arange(len(keys)),3)):
            for k in indices:
                axes[j,0].plot(steps,components[:,k],label=keys[k]);axes[j,1].plot(steps,np.cumsum(components[:,k]),label=keys[k])
            axes[j,0].set_ylabel('Signed reward components');axes[j,1].set_ylabel('Cumulative components')
            for axis in axes[j]:axis.set_xlabel('Control steps');axis.legend(fontsize=7,ncol=2);axis.axvline(len(z),color='crimson',ls=':')
        axes[-1,0].plot(steps,adjustment,'k--',label='clip / terminal override');axes[-1,1].plot(steps,np.cumsum(adjustment),'k--',label='clip / terminal override')
        axes[-1,0].legend(fontsize=7,ncol=2);axes[-1,1].legend(fontsize=7,ncol=2)
        save(fig,'reward_components')
        header=['step','interval_start_s','sample_end_s','x_m','y_m','z_m','vx_m_s','vy_m_s','vz_m_s','roll_rad','pitch_rad','yaw_rad']
        header += ['q_'+n+'_rad' for n in names]+['qd_'+n+'_rad_s' for n in names]
        header += ['action_'+n for n in ACTUATORS]+['target_'+n+('_rad_s' if i==1 else '_rad') for i,n in enumerate(ACTUATORS)]
        header += ['hip_applied_Nm','knee_applied_Nm','reward','cumulative_reward','clip_terminal_adjustment','end_code','terminated','truncated']
        header += ['reward_'+k for k in keys]+['cumulative_'+k for k in keys]
        data=np.column_stack([steps,t-dt,t,z[:,1:4],z[:,18:21],state[:,3:6],z[:,8:18],action,ctrl,effort,reward,np.cumsum(reward),adjustment,
            [r['end_code'] for r in rows],[r['terminated'] for r in rows],[r['truncated'] for r in rows],components,np.cumsum(components,axis=0)])
        assert data.shape[1]==len(header)
        with (folder/'diagnostics.csv').open('w') as f:
            w=csv.writer(f);w.writerow(header);w.writerows(data)
        np.savez_compressed(folder/'diagnostics.npz',columns=header,data=data)
        matched=min(len(z),len(old));err=z[:matched,18]-2;old_err=old[:matched,18]-2
        metric=dict(seed=seed,reset=reset,steps=len(z),duration_s=float(t[-1]),reason=reason,episode_return=float(reward.sum()),
            max_root_z_m=float(z[:,3].max()),max_root_x_m=float(z[:,1].max()),final_x_m=float(z[-1,1]),
            speed_rmse_full_episode=float(np.sqrt(np.mean((z[:,18]-2)**2))),matched_window_s=matched*dt,
            best_speed_rmse_matched=float(np.sqrt(np.mean(err**2))),original_speed_rmse_matched=float(np.sqrt(np.mean(old_err**2))),
            hip_torque_saturation_fraction=float(np.mean(np.abs(effort[:,0])>=29.999)),knee_torque_saturation_fraction=float(np.mean(np.abs(effort[:,1])>=29.999)),
            apex=endpoint['apex'],horizon_reached=len(z)==400)
        metrics.append(metric);checks.append(dict(seed=seed,rows=len(z),finite=True,action_to_target_checked=True,reward_sum_checked=True))
        group=int(endpoint['airborne_reset'])
        ax_xy[group,0].plot(z[:,1],z[:,2],label=str(seed));ax_xy[group,0].scatter(z[-1,1],z[-1,2],marker='x',color='crimson')
        ax_xy[group,1].plot(z[:,1],z[:,3],label=str(seed));ax_xy[group,1].scatter(z[-1,1],z[-1,3],marker='x',color='crimson')
        ax_rewards[group,0].plot(steps,reward,label=str(seed));ax_rewards[group,1].plot(steps,np.cumsum(reward),label=str(seed))
        links.append(f'| {seed} | {reset} | {t[-1]:.2f} | {reason} | {reward.sum():.2f} | [视频](best_rollout/{name}/replay.mp4) | [轨迹](best_rollout/{name}/trajectory.png) / [关节](best_rollout/{name}/joints.png) / [指令](best_rollout/{name}/commands.png) / [力矩](best_rollout/{name}/efforts.png) / [奖励分项](best_rollout/{name}/reward_components.png) / [CSV](best_rollout/{name}/diagnostics.csv) |')
    for group,reset in enumerate(['Ground','Airborne']):
        for j in range(2):
            ax_xy[group,j].set(xlabel='Root X (m)',ylabel='Root '+('Y' if j==0 else 'Z')+' (m)',title=reset);ax_xy[group,j].legend()
            ax_rewards[group,j].set(xlabel='Control steps',ylabel='Per-step reward' if j==0 else 'Cumulative reward',title=reset);ax_rewards[group,j].legend()
        ax_xy[group,0].set_aspect('equal',adjustable='datalim');ax_xy[group,1].axvline(3.6,color='.4',ls='--',label='Step front')
    for fig,name in [(fig_xy,'trajectories_all'),(fig_rewards,'rewards_all')]:
        fig.suptitle(f'BEST {checkpoint} | all 8 declared tests | curves stop at actual failure');fig.tight_layout(rect=[0,0,1,.96])
        for ext in ['png','pdf']:fig.savefig(out/(name+'.'+ext),dpi=160)
        plt.close(fig)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (out/'data_verification.json').write_text(json.dumps(checks,indent=2))
    repeated=json.loads((args.input/'repeatability.json').read_text())
    same=sum(r['identical'] for r in repeated)
    text=f'''# Best 模型 PhysX 仿真、视频与关节诊断

![全部轨迹](trajectories_all.png)

![每步与累计奖励](rewards_all.png)

训练已完成 **10,000,000** 步。本次固定使用 **{checkpoint}**（第 {best['training_transitions']:,} 步）；固定开发面板回报 **{best['development_mean_return']:.3f}**，属于按约25万步间隔参评候选中的最佳。未换用 last，未调参数，未续训。

重新运行原声明的4 ground＋4 airborne测试，32环境布局、完整PhysX重置、确定性Actor、1ms物理/20ms控制。本次与训练结束时保存的轨迹 **{same}/8 完全逐值一致**，即未通过跨进程逐值重复性检查，详见[重复性](best_rollout/repeatability.json)。平均回报 **{np.mean([r['episode_return'] for r in metrics]):.3f}**。全部8回合提前失败；apex只表示达到顶点事件，不能代表越障和落地恢复成功。4个ground为相同物理初态在不同环境位置的重复，不是4种随机工况。

[全部八回合视频总览](all_cases.mp4) · [原训练结果与原Actor对照](../report/INDEX.md) · [本次指标](metrics.json) · [本次实际模型审计](best_rollout/runtime_audit.json)

| seed | 初态 | 终止 s | 原始终止原因 | 回报 | 视频 | 图与数据 |
|---|---|---:|---|---:|---|---|
'''+ '\n'.join(links)+'''

每个回合均有PNG/PDF，诊断CSV/NPZ、原始traces.npz、steps.json和endpoint.json。轨迹图是车体根坐标轨迹，不是轮胎接触点轨迹；灰色为此前配对原Actor PhysX轨迹。轨迹不拼接、不延长至8秒，红叉/红线为真实失败终点。指标同时给出原Actor与best共同观察窗口的速度RMSE，避免不同时长直接比较。

关节位置与速度覆盖转向、后轮、被动前轮、髋、膝。四路输出分成归一化策略action与映射后的物理目标：转向/髋/膝为rad，后轮为rad/s。目标在[t,t+20ms)保持，测量值在区间末采集；指令用阶梯图表示。前轮为被动关节，无策略命令。

轮轴位置保留PhysX原始回绕角，因此角度图可能出现跳变；不能据此判定轮子突然反转，应同时查看实测角速度。台阶顶部为0.16m（XML半高度0.16m、中心z=0），图中地面以上高度按模型绘制。

力矩图仅画记录到的髋膝实际显式PD输入（最后1ms物理子步，±30Nm），不是整段20ms峰值。转向和后轮是PhysX原生隐式驱动，当前记录未提供其独立真实马达力矩，不能把目标值、约束反力或估算值冒充实测力矩。功率曲线为奖励使用的力矩×后步速度采样，不是积分机械功。

奖励图保留每步、累计值、全部有符号分项及裁剪/终止回报覆盖调整，因此分项可以核对到总奖励。测试没有新增人为扰动区间；airborne为初态采样。所有视频均是本次真实PhysX闭环轨迹在Isaac中的可视化回放，渲染不计算动力学；失败后冻结并明确标记，不能将冻结时长计作存活。

物理契约沿用训练：轮地各向同性0.5/min（只匹配原初态前向系数）；转向PD2.5/0.4、后轮速度增益0.005、髋膝100/6且±30Nm；模型质量保留。没有因此声称完整各向异性接触一致或sim2sim已经成功。
'''
    (out/'INDEX.md').write_text(text)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
