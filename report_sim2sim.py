"""Complete paired plots and machine-readable summary; never hide failed domains."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/sim2sim_report';OUT.mkdir(exist_ok=True)
source=json.loads((ROOT/'results/dr_solver_test/summary.json').read_text());target=json.loads((ROOT/'results/final_target_test/summary.json').read_text());names=['original','candidate'];colors=['#4167ab','#cf663b'];labels=['Original transition_4988928','DR update_0064 (not adopted)']
fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
for row in range(2):
 for k in range(2):
  if row==0:
   z=np.load(ROOT/f'results/dr_solver_test/{names[k]}/domain_00.npz')['trace'];t=z[:,0];x=z[:,1];h=z[:,3];roll=np.degrees(z[:,5])
  else:
   z=np.load(ROOT/f'results/final_target_test/nominal_{"original" if k==0 else "dr"}/traces.npz')['target'];t=z[:,0];x=z[:,1];h=z[:,3]
   from scipy.spatial.transform import Rotation
   roll=np.degrees(Rotation.from_quat(z[:,[5,6,7,4]]).as_euler('xyz')[:,0])
  for j,v in enumerate([x,h,roll]):axes[row,j].plot(t,v,color=colors[k],label=labels[k]);axes[row,j].scatter(t[-1],v[-1],color=colors[k],marker='x' if row else 'o');axes[row,j].grid(alpha=.2);axes[row,j].set_xlim(0,3);axes[row,j].set_xlabel('Time (s)')
 axes[row,0].set_ylabel('MuJoCo native' if row==0 else 'PhysX scalar mu=5');axes[row,0].axhspan(2.5,4.,alpha=.07,color='green');axes[row,1].axhline(.5,ls=':',color='black');axes[row,2].axhline(35,ls=':',color='black');axes[row,2].axhline(-35,ls=':',color='black')
for j,title in enumerate(['Forward position x (m)','Root height z (m)','Roll (degrees)']):axes[0,j].set_title(title)
axes[0,0].legend(fontsize=8);fig.suptitle('Nominal paired comparison | original action/mass/force limits | traces stop at actual endpoints')
for ext in ['png','pdf']:fig.savefig(OUT/f'nominal_comparison.{ext}',dpi=170)
plt.close(fig)
for engine in ['source','target']:
 fig,axes=plt.subplots(4,4,figsize=(14,10),constrained_layout=True)
 for d,ax in enumerate(axes.flat):
  for k in range(2):
   if engine=='source':z=np.load(ROOT/f'results/dr_solver_test/{names[k]}/domain_{d+1:02d}.npz')['trace'];t=z[:,0];v=z[:,4]
   else:z=np.load(ROOT/f'results/final_target_test/test_{d:02d}_{"original" if k==0 else "dr"}/traces.npz')['target'];t=z[:,0];v=z[:,18]
   ax.plot(t,v,label=labels[k],color=colors[k],lw=1.2);ax.plot(t[-1],v[-1],marker='x',color=colors[k],ms=5)
  mode=source[0]['cases'][d+1]['mode'];ax.set_title(f'Domain {d:02d} | {mode}',fontsize=9);ax.axhline(2,color='black',ls=':',lw=.8);ax.set_xlim(0,3);ax.grid(alpha=.2);ax.set_xlabel('s');ax.set_ylabel('vx (m/s)')
 axes[0,0].legend(fontsize=6);fig.suptitle(f'{engine}: all16 declared independent-seed test draws (4 are repeated identity controls)\nNo samples beyond true termination; 2 m/s command shown dotted')
 for ext in ['png','pdf']:fig.savefig(OUT/f'{engine}_all_domains.{ext}',dpi=160)
 plt.close(fig)
for run in ['dr_training','dr_solver_training']:
 logs=[json.loads(x) for x in (ROOT/f'results/{run}/metrics.jsonl').read_text().splitlines()];fig,axes=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
 for ax,key in zip(axes.flat,['mean_step_reward','loss','approx_kl','anchor_kl']):ax.plot([r['transitions'] for r in logs],[r[key] for r in logs]);ax.set_title(key);ax.set_xlabel('Physical control transitions');ax.grid(alpha=.2)
 fig.suptitle(run+' | stochastic training batches, not success scores')
 for ext in ['png','pdf']:fig.savefig(OUT/f'{run}.{ext}',dpi=150)
 plt.close(fig)
# Full case table plus matched common-window speed errors; unequal windows never called matched.
rows=[]
for engine in ['source','target']:
 for i in range(16):
  arrays=[]
  for k in range(2):
   if engine=='source':z=np.load(ROOT/f'results/dr_solver_test/{names[k]}/domain_{i+1:02d}.npz')['trace'];v=z[:,4];res=source[k]['cases'][i+1];end=res['end_code'];qualified=res['qualified_screen'];apex=res['apex_seen'];duration=res['duration']
   else:z=np.load(ROOT/f'results/final_target_test/test_{i:02d}_{"original" if k==0 else "dr"}/traces.npz')['target'];v=z[:,18];res=next(r for r in target if r['case']==f'test_{i:02d}_{"original" if k==0 else "dr"}');end=res['target_endpoint']['end_code'];qualified=res['target_screen_qualified'];apex=res['target_endpoint']['apex_seen'];duration=res['duration']
   arrays.append(v);rows.append(dict(engine=engine,domain=i,actor=labels[k],mode=source[0]['cases'][i+1]['mode'],duration_s=duration,end_code=end,apex_seen=apex,qualified=qualified,observed_speed_rmse_m_s=float(np.sqrt(np.mean((v-2)**2)))))
  common=min(map(len,arrays))
  for k in range(2):rows[-2+k].update(common_window_s=common*.02,common_speed_rmse_m_s=float(np.sqrt(np.mean((arrays[k][:common]-2)**2))))
with (OUT/'all_cases.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'sim2sim_accepted':False,'source_test_original_qualified':source[0]['random_domain_qualified'],'source_test_dr_qualified':source[1]['random_domain_qualified'],'source_identity_repeats':4,'source_nonidentity_original_qualified':0,'source_nonidentity_dr_qualified':1,'target_original_qualified':sum(r['target_screen_qualified'] for r in target if r['case'].startswith('test_') and r['case'].endswith('_original')),'target_dr_qualified':sum(r['target_screen_qualified'] for r in target if r['case'].startswith('test_') and r['case'].endswith('_dr')),'test_draws_per_actor':16,'adaptation_transitions_total':327680,'policy_adopted':False,'selection':'source development only, original parent unchanged','scope':'engineering; 3 second screen requires apex latch, no termination through window and x>4; not full-task certification','unresolved':['anisotropic friction vs scalar material','torsion/rolling and compliant contact','adjacent articulation self-contact behavior','stiff saturated servo integration and accelerometer sampling']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
