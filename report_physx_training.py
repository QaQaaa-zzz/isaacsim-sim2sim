"""Training curves and complete paired original/best endpoint evidence."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
out=run/'report';out.mkdir(exist_ok=True)
metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
spec=json.loads((run/'declaration.json').read_text());status=json.loads((run/'status.json').read_text());best=json.loads((run/'best_model.json').read_text())
fresh=spec.get('initialization')=='fresh'
initial_description=('Actor、106维特权Critic均按原Brax随机初始化；归一化样本数和Adam状态从零开始，未加载预训练权重。' if fresh else ('原Actor、106维特权critic与完整归一化统计初始化，复用原Brax损失/更新；新优化器（原checkpoint未保存优化器）。' if spec.get('source_ppo_port') else '原Actor初始化，新优化器/价值网络，非原Brax优化器精确续训。'))
fig,axes=plt.subplots(2,2,figsize=(11,7))
for ax,key in zip(axes.flat,['mean_step_reward','loss','kl_mean' if spec.get('source_ppo_port') else 'approx_kl','speed_rmse']):
 ax.plot([r['training_transitions'] for r in metrics],[np.nan if r[key] is None else r[key] for r in metrics]);ax.set(xlabel='Training transitions',ylabel=key);ax.grid(alpha=.3)
fig.tight_layout();fig.savefig(out/'training.png',dpi=150);fig.savefig(out/'training.pdf');plt.close(fig)
comparison=json.loads((run/'evaluation/comparison.json').read_text());table=[];links=[]
for seed in spec['best_selection']['test_seeds']:
 fig,axes=plt.subplots(2,3,figsize=(15,8));series={}
 for label in ['original','best']:
  display_label='random initialization' if fresh and label=='original' else label
  f=run/'evaluation'/f'{label}_test'/f'seed_{seed}';d=np.load(f/'traces.npz')['target'];info=json.loads((f/'steps.json').read_text());endpoint=json.loads((f/'endpoint.json').read_text());reward=np.array([x['reward'] for x in info]);t=d[:,0]
  axes[0,0].plot(d[:,1],d[:,2],label=display_label);axes[0,0].plot(d[-1,1],d[-1,2],'x')
  for ax,values in [(axes[0,1],d[:,18]),(axes[0,2],d[:,3]),(axes[1,0],np.degrees([r['roll'] for r in info])),(axes[1,1],reward),(axes[1,2],np.cumsum(reward))]:ax.plot(t,values,label=display_label);ax.plot(t[-1],values[-1],'x')
  row=dict(seed=seed,policy=label,episode_return=endpoint['return'],steps=endpoint['steps'],end_code=endpoint['end_code'],apex=endpoint['apex'],airborne_reset=endpoint['airborne_reset'],speed_rmse=float(np.sqrt(np.mean((d[:,18]-2)**2))),duration=float(t[-1]),max_z=float(np.max(d[:,3])));table.append(row)
  keys=list(info[0]['components']);csvrows=[]
  for k,ri in enumerate(info):csvrows.append(dict(t=float(t[k]),reward=ri['reward'],cumulative_reward=float(np.sum(reward[:k+1])),vx=ri['vx'],x=ri['x'],z=ri['z'],roll=ri['roll'],end_code=ri['end_code'],**{'reward_'+key:val for key,val in ri['components'].items()}))
  with (f/'steps.csv').open('w') as handle:w=csv.DictWriter(handle,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
 for ax,title in zip(axes.flat,['XY until actual endpoint','Forward speed (m/s)','Root height (m)','Roll (deg)','Per-step original reward','Cumulative original reward']):ax.set_title(title);ax.set_xlabel('Time (s)');ax.grid(alpha=.3);ax.legend()
 axes[0,0].set_xlabel('World X (m)');axes[0,0].set_ylabel('World Y (m)');axes[0,1].axhline(2,ls=':',color='k');axes[1,0].axhline(35,ls=':',color='r');axes[1,0].axhline(-35,ls=':',color='r')
 fig.suptitle(f"seed={seed}; best at {best['training_transitions']} adaptation transitions; wheel mu=.5; actual failure endpoints")
 fig.tight_layout();fig.savefig(out/f'seed_{seed}.png',dpi=140);fig.savefig(out/f'seed_{seed}.pdf');plt.close(fig);links.append(f'[seed {seed}](seed_{seed}.png)')
with (out/'cases.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
rows='\n'.join(f"| {r['seed']} | {r['policy']} | {r['episode_return']:.2f} | {r['duration']:.2f} | {r['speed_rmse']:.3f} | {r['end_code']} | {r['apex']} |" for r in table)
(out/'INDEX.md').write_text(f'''# Isaac PhysX 原奖励训练与 best 评估

已完成训练转移 {status['training_transitions']:,} / {spec['requested_training_transitions']:,}。真实动力学为 PhysX；best 是固定开发面板已评估候选（含原始初始化）中奖励最高者，不代表任务合格。

best: `{best['checkpoint']}`，开发平均原始回报 {best['development_mean_return']:.3f}。实际训练配置 [declaration.json](../declaration.json)。{initial_description} 表中及目录中的original在本轮指{'随机初始化Actor' if fresh else '原MJX Actor'}。

轮地各向同性摩擦 0.5（仅匹配原初态前进方向）；转向2.5/0.4、后轮速度反馈0.005；髋膝100/6、±30Nm显式PD，实际输入力矩用于原能耗项。物理1ms、控制20ms。质量不变，其他接触模型差异未消失。

![训练曲线](training.png)

每个面板包括XY、速度、高度、侧倾、每步奖励和累计奖励。图在真实终止处截止；4个ground种子产生相同确定性初态，不能视为4个独立训练种子。4个airborne初态由冻结种子决定；开发和测试seed分开，测试不用于选择best。

{' · '.join(links)}

| seed | policy | return | duration s | speed RMSE m/s | end_code | apex |
|---|---|---:|---:|---:|---|---|
{rows}

原始逐步数据及分项CSV在 `evaluation/original_test` 和 `evaluation/best_test`。视频是这些真实PhysX轨迹的标注回放，渲染不执行物理，终止后冻结姿态。仅展示首个ground/airborne种子的预声明代表视频；全部8个条件都保留图、数据和终点。
''')
print('REPORT',out/'INDEX.md')
