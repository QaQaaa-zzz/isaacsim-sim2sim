"""Finite-budget native PhysX PPO adaptation with explicit provenance and best evaluation."""
import argparse,os,sys,json,time,traceback,hashlib,math
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMNI_KIT_ACCEPT_EULA']='YES';sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent

def write_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def main(args):
    spec=json.loads(args.config.read_text());ppo=spec['ppo'];out=args.output;out.mkdir(parents=True,exist_ok=False)
    if args.transitions is not None:spec['requested_training_transitions']=args.transitions
    if args.num_envs is not None:ppo['num_envs']=args.num_envs
    spec['engineering_smoke']=bool(args.transitions is not None)
    write_json(out/'declaration.json',spec);write_json(out/'status.json',dict(stage='initializing',completed=False,training_transitions=0,pid=os.getpid()))
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'limit_cpu_threads':4,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
    try:
        import numpy as np,torch
        from torch import nn
        from torch.utils.tensorboard import SummaryWriter
        from train_dr import Network
        from physx_task import PhysxTask
        from physx_training_math import gae_returns
        from policy_runtime import Actor
        torch.set_num_threads(ppo['torch_threads']);torch.manual_seed(ppo['seed'])
        env=PhysxTask(spec,ppo['num_envs'],ppo['seed']);write_json(out/'runtime_audit.json',env.audit)
        net=Network(ROOT/spec['initial_actor']);normalizer_mean=net.mean.clone();normalizer_std=net.std.clone()
        parity=np.load(ROOT/'policy/reference_inference.npz')
        with torch.no_grad():error=float(np.max(np.abs(torch.tanh(net.distribution(torch.tensor(parity['observations'])).loc).numpy()-parity['actions'])))
        assert error<3e-5;write_json(out/'initial_actor_parity.json',dict(max_abs_error=error,threshold=3e-5))
        optimizer=torch.optim.Adam([{'params':net.actor.parameters(),'lr':ppo['actor_lr']},{'params':net.critic.parameters(),'lr':ppo['critic_lr']}])
        tb=SummaryWriter(str(out/'tensorboard'));metrics=(out/'metrics.jsonl').open('a');episodes=(out/'episodes.jsonl').open('a')
        total=0;updates=0;eval_physics=0;eval_recorded=0;initial_t=time.monotonic();next_eval=spec['best_selection']['eval_interval_training_transitions'];best=None
        identity=json.loads((ROOT/'policy/identity.json').read_text())
        def save_checkpoint(label):
            path=out/'checkpoints'/label;path.mkdir(parents=True,exist_ok=True);net.export(path/'actor.npz')
            torch.save({'network':net.state_dict(),'optimizer':optimizer.state_dict(),'training_transitions':total,'updates':updates,'spec':spec,'torch_rng':torch.get_rng_state()},path/'training_state.pt')
            write_json(path/'identity.json',dict(kind='derived_PhysX_PPO_adaptation',parent=identity,training_transitions=total,source_checkpoint_transitions=4988928,actor_sha256=hashlib.sha256((path/'actor.npz').read_bytes()).hexdigest(),exact_environment_resume=False))
            (path/'resolved_config.json').write_bytes((ROOT/'policy/resolved_config.json').read_bytes())
            return path
        def evaluate(actor,seeds,folder):
            nonlocal eval_physics,eval_recorded
            folder.mkdir(parents=True,exist_ok=False);count=len(seeds);assert count<=env.n
            env.hard_reset();env.reset(np.arange(count),seeds=seeds)
            active=np.ones(count,bool);traces=[[] for _ in seeds];info_rows=[[] for _ in seeds];endpoints=[None]*count
            for k in range(spec['episode_horizon']):
                action=actor(env.obs);obs,reward,done,info=env.step(action);eval_physics+=env.n
                for i in range(count):
                    if active[i]:
                        row=env.trace[i].copy();row[0]=(k+1)*spec['control_dt'];traces[i].append(row);info_rows[i].append(info[i]);eval_recorded+=1
                        if done[i]:endpoints[i]=info[i];active[i]=False
                if not active.any():break
                env.reset(np.flatnonzero(done))
            summary=[]
            for i,seed in enumerate(seeds):
                endpoint=endpoints[i] or info_rows[i][-1];path=folder/f'seed_{seed}';path.mkdir()
                np.savez_compressed(path/'traces.npz',target=traces[i],joint_names=env.names)
                write_json(path/'case.json',dict(seed=seed,recorded_engine='PhysX',contact_label='Wheel friction 0.5; calibrated PD; actual endpoint',friction=.5,duration=len(traces[i])*spec['control_dt']))
                write_json(path/'steps.json',info_rows[i]);write_json(path/'endpoint.json',endpoint)
                summary.append(dict(seed=seed,**endpoint,horizon_reached=len(traces[i])==spec['episode_horizon']))
            write_json(folder/'summary.json',summary);env.reset(np.arange(env.n))
            return float(np.mean([r['return'] for r in summary])),summary
        def score_development():
            nonlocal best
            checkpoint=save_checkpoint(f'transition_{total:08d}');score,summary=evaluate(Actor(checkpoint/'actor.npz'),spec['best_selection']['seeds'],out/'development'/f'transition_{total:08d}')
            tb.add_scalar('development/mean_episode_return',score,total)
            tb.add_scalar('development/apex_fraction',np.mean([r['apex'] for r in summary]),total)
            tb.add_scalar('development/physical_failure_fraction',np.mean([r['end_code'] in [2,3,4,5,6] for r in summary]),total)
            if best is None or score>best['development_mean_return']:
                best=dict(checkpoint=str(checkpoint.resolve()),training_transitions=total,development_mean_return=score,selection='fixed 4-ground/4-airborne development mean original episode return; earlier wins tie',qualified=False)
                write_json(out/'best_model.json',best)
            return score
        write_json(out/'status.json',dict(stage='initial_development',completed=False,training_transitions=0,pid=os.getpid()))
        score_development();net.train();env.reset(np.arange(env.n));last_status={}
        budget=spec['requested_training_transitions'];assert budget%env.n==0,'Budget must be divisible by vector width'
        while total<budget:
            unroll=min(ppo['unroll'],(budget-total)//env.n);buffer=[];sample_start=time.monotonic();info_batch=[]
            for t in range(unroll):
                observation=torch.tensor(env.obs.copy())
                with torch.no_grad():
                    distribution=net.distribution(observation);latent=distribution.sample();action=torch.tanh(latent);logp=distribution.log_prob(latent).sum(-1);value=net.value(observation)
                obs,reward,done,info=env.step(action.numpy());total+=env.n
                for i in np.flatnonzero(done):episodes.write(json.dumps(dict(training_transitions=total,env=int(i),**info[i]))+'\n')
                info_batch.extend(info)
                buffer.append((observation,latent,logp,value,reward*ppo['reward_scale'],done))
                env.reset(np.flatnonzero(done))
            sample_s=time.monotonic()-sample_start
            with torch.no_grad():last_value=net.value(torch.tensor(env.obs.copy())).numpy()
            advantages,returns=gae_returns(np.stack([b[4] for b in buffer]),np.stack([b[3].numpy() for b in buffer]),np.stack([b[5] for b in buffer]),last_value,gamma=ppo['gamma'],lam=ppo['gae_lambda'])
            obs=torch.cat([b[0] for b in buffer]);latent=torch.cat([b[1] for b in buffer]);oldlog=torch.cat([b[2] for b in buffer]);adv=torch.tensor(advantages.flatten());ret=torch.tensor(returns.flatten());adv=(adv-adv.mean())/(adv.std()+1e-8)
            optimize_start=time.monotonic();losses=[];kls=[];gradnorms=[];kl_stop=False
            for epoch in range(ppo['epochs']):
                for ix in torch.randperm(len(obs)).split(ppo['minibatch']):
                    distribution=net.distribution(obs[ix]);logp=distribution.log_prob(latent[ix]).sum(-1);logratio=logp-oldlog[ix];ratio=logratio.exp()
                    policy_loss=-torch.minimum(ratio*adv[ix],ratio.clamp(1-ppo['clip'],1+ppo['clip'])*adv[ix]).mean()
                    value_loss=(net.value(obs[ix])-ret[ix]).square().mean();entropy=distribution.entropy().sum(-1).mean()
                    loss=policy_loss+ppo['value_coeff']*value_loss-ppo['entropy_coeff']*entropy
                    kl=float(((ratio-1)-logratio).mean().detach())
                    if not np.isfinite(float(loss.detach())) or not np.isfinite(kl):raise RuntimeError('Nonfinite PPO objective')
                    if kl>ppo['max_kl']:kl_stop=True;break
                    optimizer.zero_grad();loss.backward();gn=nn.utils.clip_grad_norm_(net.parameters(),ppo['max_grad_norm']);optimizer.step()
                    losses.append(float(loss.detach()));kls.append(kl);gradnorms.append(float(gn))
                if kl_stop:break
            updates+=1;assert torch.equal(net.mean,normalizer_mean) and torch.equal(net.std,normalizer_std)
            record=dict(stage='training',completed=False,pid=os.getpid(),training_transitions=total,budget=budget,update=updates,sampling_s=sample_s,optimization_s=time.monotonic()-optimize_start,wall_s=time.monotonic()-initial_t,sampling_transitions_per_s=unroll*env.n/sample_s,mean_step_reward=float(np.mean([i['reward'] for i in info_batch])),loss=float(np.mean(losses)) if losses else None,approx_kl=float(np.mean(kls)) if kls else None,kl_stop=kl_stop,accepted_minibatches=len(losses),speed_rmse=float(np.sqrt(np.mean([(i['vx']-2)**2 for i in info_batch]))),evaluation_physics_transitions=eval_physics,evaluation_recorded_transitions=eval_recorded,best=best)
            metrics.write(json.dumps(record)+'\n');metrics.flush();episodes.flush()
            for key in ['mean_step_reward','loss','approx_kl','speed_rmse','sampling_transitions_per_s','accepted_minibatches']:
                if record[key] is not None:tb.add_scalar('training/'+key,record[key],total)
            tb.flush();write_json(out/'status.json',record);last_status=record;print('UPDATE',json.dumps(record),flush=True)
            if total>=next_eval or total==budget:
                write_json(out/'status.json',dict(record,stage='development'));score_development();next_eval=(total//spec['best_selection']['eval_interval_training_transitions']+1)*spec['best_selection']['eval_interval_training_transitions']
            elif updates%ppo['save_every_updates']==0:save_checkpoint('latest')
        write_json(out/'status.json',dict(last_status,stage='final_evaluation',best=best));save_checkpoint('last')
        best_score,best_summary=evaluate(Actor(Path(best['checkpoint'])/'actor.npz'),spec['best_selection']['test_seeds'],out/'evaluation'/'best_test')
        baseline_score,baseline_summary=evaluate(Actor(ROOT/spec['initial_actor']),spec['best_selection']['test_seeds'],out/'evaluation'/'original_test')
        write_json(out/'evaluation'/'comparison.json',dict(best=best,best_test_mean_return=best_score,original_test_mean_return=baseline_score,best_test=best_summary,original_test=baseline_summary,testing_role='declared paired engineering test; ground seeds repeat same deterministic reset; airborne seeds differ'))
        write_json(out/'status.json',dict(last_status,stage='training_and_evaluation_complete',completed=True,best=best,evaluation_physics_transitions=eval_physics,evaluation_recorded_transitions=eval_recorded,wall_s=time.monotonic()-initial_t,media_pending=True))
        tb.close();metrics.close();episodes.close()
    except BaseException:
        write_json(out/'error.json',dict(traceback=traceback.format_exc(),pid=os.getpid()))
        state=json.loads((out/'status.json').read_text());write_json(out/'status.json',dict(state,stage='failed',completed=False,error='See error.json'));raise
    finally:app.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--transitions',type=int);parser.add_argument('--num-envs',type=int);main(parser.parse_args())
