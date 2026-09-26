"""Bounded CPU PPO engineering adaptation. Frozen normalizer and original reward.
A new optimizer/critic is explicitly used; this is not an exact Brax resume.
"""
import os,sys,json,time,argparse,hashlib
os.environ['JAX_PLATFORMS']='cpu';os.environ['OPENBLAS_NUM_THREADS']='1';sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.distributions import Normal
from source_task import SourceTask,ROOT
from domains import sample_domain

class Network(nn.Module):
 def __init__(self,path):
  super().__init__();p=dict(np.load(path));self.register_buffer('mean',torch.tensor(p['mean']));self.register_buffer('std',torch.tensor(p['std']));self.actor=nn.ModuleList([nn.Linear(*p[f'hidden_{i}_kernel'].shape) for i in range(4)]);self.critic=nn.Sequential(nn.Linear(76,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1))
  with torch.no_grad():
   for i,l in enumerate(self.actor):l.weight.copy_(torch.tensor(p[f'hidden_{i}_kernel'].T));l.bias.copy_(torch.tensor(p[f'hidden_{i}_bias']))
 def distribution(self,obs):
  x=(obs-self.mean)/self.std
  for i,l in enumerate(self.actor):x=l(x);x=nn.functional.silu(x) if i<3 else x
  loc,scale=x.chunk(2,dim=-1);return Normal(loc,nn.functional.softplus(scale)+.001)
 def value(self,obs):return self.critic(torch.clamp((obs-self.mean)/self.std,-20,20)).squeeze(-1)
 def export(self,path):
  arrays={'mean':self.mean.detach().numpy(),'std':self.std.detach().numpy()}
  for i,l in enumerate(self.actor):arrays[f'hidden_{i}_kernel']=l.weight.detach().numpy().T;arrays[f'hidden_{i}_bias']=l.bias.detach().numpy()
  np.savez(path,**arrays)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text());out=a.output;out.mkdir(parents=True,exist_ok=False);(out/'declaration.json').write_text(json.dumps(cfg,indent=2));spec=json.loads((ROOT/cfg['domain_config']).read_text());(out/'domain_randomization.json').write_text(json.dumps(spec,indent=2));torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
 net=Network(ROOT/'policy/actor.npz');frozen=Network(ROOT/'policy/actor.npz');frozen.load_state_dict(net.state_dict());frozen.requires_grad_(False)
 check=np.load(ROOT/'policy/reference_inference.npz');error=np.max(np.abs(torch.tanh(net.distribution(torch.tensor(check['observations'])).loc).detach().numpy()-check['actions']));assert error<3e-5
 (out/'initial_actor_parity.json').write_text(json.dumps({'max_abs_error':float(error),'threshold':3e-5}))
 optimizer=torch.optim.Adam([{'params':net.actor.parameters(),'lr':cfg['actor_lr']},{'params':net.critic.parameters(),'lr':cfg['critic_lr']}]);N=cfg['num_envs'];T=cfg['unroll'];envs=[SourceTask() for _ in range(N)];episodes=0;manifest=open(out/'domains.jsonl','w');episode_log=open(out/'episodes.jsonl','w');metrics=open(out/'metrics.jsonl','w');t0=time.time()
 def reset(e):
  nonlocal episodes
  seed=cfg['seed']+episodes;d=sample_domain(spec,seed,'train');d['episode']=episodes;manifest.write(json.dumps(d)+'\n');episodes+=1;return e.reset(d,seed)
 obs=np.stack([reset(e) for e in envs]);running_return=np.zeros(N);running_steps=np.zeros(N,int);last_infos=[{} for _ in range(N)];updates=cfg['transitions']//(N*T)
 for update in range(updates):
  buf=[];sampling_start=time.time();net.eval()
  for t in range(T):
   ot=torch.tensor(obs)
   with torch.no_grad():dist=net.distribution(ot);latent=dist.sample();act=torch.tanh(latent);logp=dist.log_prob(latent).sum(-1);value=net.value(ot)
   rewards=[];done=[];nextobs=[]
   for i,e in enumerate(envs):
    o,r,d,info=e.step(act[i].numpy());running_return[i]+=r;running_steps[i]+=1;last_infos[i]=info
    if d:
     episode_log.write(json.dumps(dict(update=update,env=i,steps=int(running_steps[i]),return_=float(running_return[i]),domain=e.domain,endpoint=info))+'\n');running_return[i]=0;running_steps[i]=0;o=reset(e)
    rewards.append(r*cfg['reward_scale']);done.append(d);nextobs.append(o)
   buf.append((ot,latent,logp,value,torch.tensor(rewards,dtype=torch.float32),torch.tensor(done,dtype=torch.float32)));obs=np.stack(nextobs)
  sample_s=time.time()-sampling_start
  with torch.no_grad():nv=net.value(torch.tensor(obs))
  adv=torch.zeros((T,N));gae=torch.zeros(N)
  for t in reversed(range(T)):
   _,_,_,v,r,d=buf[t];mask=1-d;delta=r+cfg['gamma']*nv*mask-v;gae=delta+cfg['gamma']*cfg['gae_lambda']*mask*gae;adv[t]=gae;nv=v
  Bobs=torch.cat([b[0] for b in buf]);Blatent=torch.cat([b[1] for b in buf]);Blogp=torch.cat([b[2] for b in buf]);Bvalue=torch.cat([b[3] for b in buf]);Badv=adv.flatten();returns=Badv+Bvalue;Badv=(Badv-Badv.mean())/(Badv.std()+1e-8);size=len(Bobs);klvalues=[];losses=[];anchorvalues=[];opt_start=time.time();net.train();early=False
  for epoch in range(cfg['epochs']):
   for ix in torch.randperm(size).split(cfg['minibatch']):
    dist=net.distribution(Bobs[ix]);lp=dist.log_prob(Blatent[ix]).sum(-1);ratio=(lp-Blogp[ix]).exp();policy=-torch.minimum(ratio*Badv[ix],ratio.clamp(1-cfg['clip'],1+cfg['clip'])*Badv[ix]).mean();v=net.value(Bobs[ix]);vloss=(v-returns[ix]).square().mean()
    with torch.no_grad():old=frozen.distribution(Bobs[ix])
    anchor=torch.distributions.kl_divergence(old,dist).sum(-1).mean();entropy=dist.entropy().sum(-1).mean();loss=policy+cfg['value_coeff']*vloss+cfg['anchor_kl']*anchor-cfg['entropy_coeff']*entropy;optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),cfg['max_grad_norm']);optimizer.step()
    kl=float(((ratio-1)-(lp-Blogp[ix])).mean().detach());klvalues.append(kl);losses.append(float(loss.detach()));anchorvalues.append(float(anchor.detach()))
    if not np.isfinite(losses[-1]):raise RuntimeError('nonfinite PPO loss')
    if kl>cfg['max_kl']:early=True;break
   if early:break
  record=dict(update=update+1,transitions=(update+1)*N*T,sampling_s=sample_s,optimize_s=time.time()-opt_start,wall_s=time.time()-t0,mean_step_reward=float(torch.stack([b[4] for b in buf]).mean()/cfg['reward_scale']),loss=float(np.mean(losses)),approx_kl=float(np.mean(klvalues)),anchor_kl=float(np.mean(anchorvalues)),kl_stop=early,episodes=episodes,completed=False)
  metrics.write(json.dumps(record)+'\n');metrics.flush();manifest.flush();episode_log.flush();(out/'status.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
  if (update+1)%cfg['save_every']==0 or update+1==updates:
   ck=out/f'update_{update+1:04d}';ck.mkdir();net.export(ck/'actor.npz');torch.save({'network':net.state_dict(),'optimizer':optimizer.state_dict(),'config':cfg,'update':update+1},ck/'training_state.pt')
 record['completed']=True;(out/'status.json').write_text(json.dumps(record,indent=2));manifest.close();metrics.close();episode_log.close()
if __name__=='__main__':main()
