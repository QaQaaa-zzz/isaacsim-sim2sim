"""Original Brax PPO update semantics with independent real-PhysX sampling/eval."""
import argparse,hashlib,json,logging,os,resource,subprocess,sys,time,traceback,faulthandler,signal
from pathlib import Path
os.environ.update(JAX_PLATFORMS='cpu',OMNI_KIT_ACCEPT_EULA='YES',OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent

def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--engineering-blocks',type=int);a=parser.parse_args()
    spec=json.loads(a.config.read_text());ppo=spec['ppo'];n=ppo['num_parallel_envs'];t=ppo['unroll_length'];block=n*t
    assert n==ppo['batch_size']*ppo['num_minibatches']
    budget=spec['requested_training_transitions'] if a.engineering_blocks is None else a.engineering_blocks*block
    assert budget%block==0 and budget>0
    if a.engineering_blocks is None:assert spec['launch_ready'],'Formal launch requires verified engineering gate'
    spec['engineering_smoke']=a.engineering_blocks is not None;spec['requested_training_transitions']=budget
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);write(out/'declaration.json',spec)
    snapshot=out/'implementation';snapshot.mkdir();manifest={}
    for name in ['train_physx_brax.py','brax_physx_learner.py','physx_evaluation_worker.py','physx_task.py','policy_runtime.py','source_reset.py','usd_model.py','physx_training_math.py','collision_geometry.py','drive_calibration.py','render_traces.py','scene_visuals.py','report_physx_training.py']:
        data=(ROOT/name).read_bytes();(snapshot/name).write_bytes(data);manifest[name]=hashlib.sha256(data).hexdigest()
    write(out/'implementation_hashes.json',manifest)
    status=dict(stage='initializing',completed=False,pid=os.getpid(),training_transitions=0,budget=budget,update=0)
    faulthandler.enable();faulthandler.register(signal.SIGUSR1,all_threads=False)
    write(out/'status.json',status);queue=out/'evaluation_queue';queue.mkdir();worker=None;app=None;tb=None;started=time.monotonic()
    try:
        worker_log=(out/'evaluation_worker.log').open('w')
        worker=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(ROOT/'physx_evaluation_worker.py'),'--config',str(out/'declaration.json'),'--queue',str(queue)],cwd=ROOT,stdout=worker_log,stderr=subprocess.STDOUT)
        def wait_for(path,timeout=600):
            start=time.monotonic()
            while not path.exists():
                if worker.poll() is not None:raise RuntimeError('Evaluation worker exited; see evaluation_worker.log')
                if time.monotonic()-start>timeout:raise TimeoutError(str(path))
                time.sleep(.25)
            return json.loads(path.read_text())
        wait_for(queue/'ready.json')
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'limit_cpu_threads':4,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
        faulthandler.register(signal.SIGUSR1,all_threads=False)
        def stage(name):
            status.update(stage=name,wall_s=time.monotonic()-started);write(out/'status.json',status);print('STAGE',name,flush=True)
        logging.getLogger('jax').setLevel(logging.WARNING)
        stage('importing_jax')
        import jax
        import numpy as np
        stage('importing_brax')
        from brax.training import types
        stage('importing_tensorboard')
        from tensorboardX import SummaryWriter
        stage('importing_learner')
        from brax_physx_learner import Learner
        stage('importing_environment')
        from physx_task import PhysxTask
        from policy_runtime import Actor
        from source_reset import initial_env_keys
        stage('loading_checkpoint');learner=Learner(Path(spec['initial_checkpoint']),ppo)
        np.testing.assert_array_equal(learner.initial_env_keys(n),initial_env_keys(ppo['seed'],n))
        stage('building_physics');env=PhysxTask(spec,n,ppo['seed']);write(out/'runtime_audit.json',env.audit);write(out/'learner_identity.json',learner.runtime_identity)
        tb=SummaryWriter(str(out/'tensorboard'));log=(out/'metrics.jsonl').open('a');episodes=(out/'episodes.jsonl').open('a');best=None;eval_count=0;eval_transitions=0
        def checkpoint(label,total):
            path=out/'checkpoints'/label;learner.save(path,total);(path/'resolved_config.json').write_bytes((ROOT/'policy/resolved_config.json').read_bytes())
            obs={'state':env.obs.copy(),'privileged_state':env.privileged_obs.copy()}
            error=float(np.max(np.abs(Actor(path/'actor.npz')(obs['state'])-learner.deterministic(obs))))
            if error>3e-5:raise RuntimeError(f'Actor export mismatch: {error}')
            write(path/'export_verification.json',dict(max_action_error=error,passed=True));return path
        def evaluate(path,seeds,folder):
            nonlocal eval_count,eval_transitions
            response=queue/'response.json'
            if response.exists():response.unlink()
            eval_count+=1;write(queue/'request.json',dict(id=str(eval_count),actor_path=str(path/'actor.npz'),seeds=seeds,output=str(folder),checkpoint_label=path.name))
            reply=wait_for(response)
            if str(reply['id'])!=str(eval_count) or not reply.get('complete'):raise RuntimeError(f'Evaluation failed: {reply}')
            eval_transitions+=reply['physics_transitions'];return reply
        def score(total):
            nonlocal best
            path=checkpoint(f'transition_{total:08d}',total)
            result=evaluate(path,spec['best_selection']['seeds'],out/'development'/path.name)
            value=result['mean_return'];tb.add_scalar('development/mean_episode_return',value,total)
            tb.add_scalar('development/failure_fraction',np.mean([s['terminated'] for s in result['summary']]),total)
            tb.add_scalar('development/mean_duration_s',np.mean([s['steps']*.02 for s in result['summary']]),total)
            if best is None or value>best['development_mean_return']:
                best=dict(checkpoint=str(path),training_transitions=total,development_mean_return=value,selection='all completed source PPO blocks plus initialization; fixed eight-case development return; earlier wins tie',qualified=False)
                write(out/'best_model.json',best)
            return result
        status.update(stage='initial_development');write(out/'status.json',status);initial=score(0)
        # Repeated scoring is separate from training; reject only numerical corruption,
        # report repeatability rather than silently replacing either result.
        if a.engineering_blocks is not None:
            repeated=evaluate(Path(best['checkpoint']),spec['best_selection']['seeds'],out/'engineering_repeat')
            differences=[]
            for seed in spec['best_selection']['seeds']:
                first=np.load(out/'development/transition_00000000'/f'seed_{seed}'/'traces.npz')['target'];second=np.load(out/'engineering_repeat'/f'seed_{seed}'/'traces.npz')['target']
                differences.append(dict(seed=seed,identical=bool(np.array_equal(first,second)),max_abs_difference=float(np.max(np.abs(first-second))) if first.shape==second.shape else None))
            write(out/'repeatability.json',differences)
        for update in range(1,budget//block+1):
            learner.begin_rollout();batch=[];info_batch=[];sample_start=time.monotonic()
            for tick in range(t):
                obs={'state':env.obs.copy(),'privileged_state':env.privileged_obs.copy()};action,extra=learner.sample(obs)
                _,reward,done,info=env.step(action);total=(update-1)*block+(tick+1)*n
                for i in np.flatnonzero(done):episodes.write(json.dumps(dict(training_transitions=total,env=int(i),**info[i]))+'\n')
                info_batch.extend(info);env.reset(np.flatnonzero(done))
                next_obs={'state':env.obs.copy(),'privileged_state':env.privileged_obs.copy()}
                batch.append(types.Transition(obs,action,reward,1-done.astype(np.float32),next_obs,dict(policy_extras=extra,state_extras=dict(truncation=np.zeros(n,np.float32),time_out=np.zeros(n,np.float32)))))
                if tick%8==0 or tick==t-1:
                    status.update(stage='sampling',training_transitions=total,update=update-1,sampling_tick=tick+1,best=best,wall_s=time.monotonic()-started)
                    write(out/'status.json',status);episodes.flush()
            sampling_s=time.monotonic()-sample_start;status.update(stage='optimizing');write(out/'status.json',status)
            data=jax.tree.map(lambda *x:np.swapaxes(np.stack(x),0,1),*batch);opt_start=time.monotonic();metrics=learner.update(data);opt_s=time.monotonic()-opt_start
            record=dict(metrics,stage='training',completed=False,pid=os.getpid(),training_transitions=total,budget=budget,update=update,mean_step_reward=float(np.mean([i['reward'] for i in info_batch])),
                loss=metrics['total_loss'],approx_kl=metrics.get('approx_kl',metrics.get('kl_estimate')),speed_rmse=float(np.sqrt(np.mean([(i['vx']-2)**2 for i in info_batch]))),
                sampling_s=sampling_s,optimization_s=opt_s,sampling_transitions_per_s=block/sampling_s,wall_s=time.monotonic()-started,max_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                evaluation_physics_transitions=eval_transitions,best=best)
            log.write(json.dumps(record)+'\n');log.flush();episodes.flush()
            for k,v in record.items():
                if isinstance(v,(float,int)) and not isinstance(v,bool) and k not in ['pid']:tb.add_scalar('training/'+k,v,total)
            tb.flush();status=record;write(out/'status.json',status);print('UPDATE',json.dumps(record),flush=True)
            status.update(stage='development');write(out/'status.json',status);score(total);status.update(best=best)
        checkpoint('last',budget)
        status.update(stage='final_evaluation',best=best);write(out/'status.json',status)
        best_result=evaluate(Path(best['checkpoint']),spec['best_selection']['test_seeds'],out/'evaluation/best_test')
        original_result=evaluate(out/'checkpoints/transition_00000000',spec['best_selection']['test_seeds'],out/'evaluation/original_test')
        write(out/'evaluation/comparison.json',dict(best=best,best_test_mean_return=best_result['mean_return'],original_test_mean_return=original_result['mean_return'],best_test=best_result['summary'],original_test=original_result['summary'],testing_role='fixed paired eight-case PhysX engineering test; source Actor versus all-block reward-best'))
        status.update(stage='training_and_evaluation_complete',completed=True,best=best,evaluation_physics_transitions=eval_transitions,media_pending=True,wall_s=time.monotonic()-started)
        write(out/'status.json',status);tb.close();tb=None;log.close();episodes.close()
    except BaseException:
        write(out/'error.json',dict(traceback=traceback.format_exc(),pid=os.getpid()));status.update(stage='failed',completed=False,error='See error.json');write(out/'status.json',status);raise
    finally:
        (queue/'stop').touch()
        if worker is not None:
            try:worker.wait(timeout=30)
            except subprocess.TimeoutExpired:worker.terminate()
        if tb is not None:tb.close()
        if app is not None:app.close()

if __name__=='__main__':main()
