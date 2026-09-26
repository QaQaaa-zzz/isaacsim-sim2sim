"""Paired domain evaluation and source-only candidate selection."""
import argparse,json,time
from pathlib import Path
import numpy as np
from source_task import SourceTask,ROOT
from domains import sample_domain,identity_domain
from policy_runtime import Actor
p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--role',choices=['dev','test'],default='dev');p.add_argument('--candidate',type=Path);p.add_argument('--duration',type=float,default=3.);a=p.parse_args();assert a.duration<=4;a.output.mkdir(parents=True,exist_ok=False)
spec=json.loads((a.training/'domain_randomization.json').read_text());domains=[identity_domain()]+[sample_domain(spec,i,a.role) for i in range(16)];(a.output/'domains.json').write_text(json.dumps(domains,indent=2));actors={'original':ROOT/'policy/actor.npz'}
if a.candidate:actors['candidate']=a.candidate
else:actors.update({x.parent.name:x for x in sorted(a.training.glob('update_*/actor.npz'))})
summary=[]
for name,path in actors.items():
 actor=Actor(path);env=SourceTask();dest=a.output/name;dest.mkdir();results=[]
 for i,domain in enumerate(domains):
  o=env.reset(domain,i);rows=[];done=False;trace_obs=[]
  for step in range(round(a.duration/.02)):
   act=actor(o);trace_obs.append(o.copy());o,r,done,info=env.step(act);rows.append(np.r_[(step+1)*.02,info['x'],env.d.qpos[1],info['z'],info['vx'],info['roll'],info['pitch'],info['yaw'],r,info['apex'],done,info['end_code'],env.d.qpos.copy(),act,*info['components'].values()])
   if done:break
  z=np.array(rows);np.savez_compressed(dest/f'domain_{i:02d}.npz',trace=z,obs=trace_obs,component_names=list(info['components']))
  res={'domain':i,'mode':domain['contact_mode'],'duration':len(z)*.02,'end_code':info['end_code'],'apex_seen':info['apex'],'completed_window':not done,'qualified_screen':bool(not done and info['apex'] and info['x']>4.),'speed_rmse':float(np.sqrt(np.mean((z[:,4]-2)**2))),'max_roll_deg':float(np.degrees(np.max(np.abs(z[:,5])))),'max_z':float(z[:,3].max()),'end_x':info['x'],'return':float(z[:,8].sum())};results.append(res)
 s={'actor':name,'actor_path':str(path),'nominal':results[0],'random_domain_qualified':sum(x['qualified_screen'] for x in results[1:]),'random_domain_apex':sum(x['apex_seen'] for x in results[1:]),'random_domain_completed':sum(x['completed_window'] for x in results[1:]),'random_domain_count':len(results)-1,'random_mean_speed_rmse':float(np.mean([x['speed_rmse'] for x in results[1:]])),'cases':results};summary.append(s);print(json.dumps({k:v for k,v in s.items() if k!='cases'}),flush=True)
(a.output/'summary.json').write_text(json.dumps(summary,indent=2))
if a.role=='dev':
 base=summary[0];eligible=[s for s in summary[1:] if s['nominal']['qualified_screen'] and s['nominal']['speed_rmse']<=base['nominal']['speed_rmse']+.2];selected=max(eligible,key=lambda s:(s['random_domain_qualified'],-s['random_mean_speed_rmse'])) if eligible else base
 result={'selected_actor':selected['actor'],'actor_path':selected['actor_path'],'eligible_count':len(eligible),'criterion':'nominal screen safe apex and speed RMSE <= original +0.2 m/s; then largest random-domain safe apex completion, tie lower speed RMSE','adopted':False,'reason':'source-only engineering selection; PhysX and disjoint test pending'};(a.output/'selection.json').write_text(json.dumps(result,indent=2));print('SELECTION',json.dumps(result),flush=True)
