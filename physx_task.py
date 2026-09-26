"""Real PhysX vector environment. MuJoCo is used ONLY for model/kinematics.

Initialize SimulationApp before importing this module. Native rear/steering
drives and explicit capped hip/knee PD preserve known motor work in the reward.
"""
import json,sys,os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
import omni.usd
from pxr import UsdGeom,UsdPhysics,PhysxSchema,Sdf
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.cloner import Cloner
from usd_model import build
from policy_runtime import Observation,control
from physx_training_math import pd_effort,continuous_joint_position
sys.dont_write_bytecode=True
os.environ.setdefault('JAX_PLATFORMS','cpu')
sys.path.insert(0,'/home/qy/DVGC/JIT/src')
from jit_dvgc.config import load_config
from jit_dvgc import rewards,semantics
rewards.jp=np;semantics.jp=np
ROOT=Path(__file__).resolve().parent
RAW=json.loads((ROOT/'policy/resolved_config.json').read_text())
CFG=load_config(ROOT/'policy/resolved_config.json',runtime_only=True)

class PhysxTask:
    def __init__(self,spec,n,seed):
        self.spec=spec;self.n=n;self.rng=[np.random.default_rng(seed+i) for i in range(n)];self.source_port=spec.get('source_ppo_port',False)
        self.dt=spec['physics_dt'];self.substeps=round(spec['control_dt']/self.dt)
        self.world=World(physics_dt=self.dt,rendering_dt=.02,backend='numpy')
        self.world.get_physics_context().enable_fabric(False)
        if self.source_port:
            context=self.world.get_physics_context()
            scene=context.get_current_physics_scene_prim()
            PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableEnhancedDeterminismAttr(True)
        stage=omni.usd.get_context().get_stage();UsdGeom.Scope.Define(stage,'/World/envs');UsdGeom.Xform.Define(stage,'/World/envs/env_0')
        self.m,conv=build(stage,ROOT/'model/source.xml',root_path='/World/envs/env_0/Bike',target_drive_gains=spec['target_drive_gains'])
        # Move obstacle into each environment; keep one infinite common floor.
        step_id=self.m.geom('step').id;old=f'/World/step_g{step_id}';new='/World/envs/env_0/step'
        Sdf.CopySpec(stage.GetRootLayer(),old,stage.GetRootLayer(),new);stage.RemovePrim(old)
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.MaterialAPI):
                api=UsdPhysics.MaterialAPI(prim)
                # All contacts involving a wheel use .5 with min combine, including floor/step.
                name=prim.GetName()
                g=int(name.split('_')[-1]) if name.startswith('material_') else -1
                if g>=0 and 'wheel' in self.m.geom(g).name:
                    api.CreateStaticFrictionAttr(spec['contact_required']['static_friction']);api.CreateDynamicFrictionAttr(spec['contact_required']['dynamic_friction'])
                PhysxSchema.PhysxMaterialAPI.Apply(prim).CreateFrictionCombineModeAttr('min')
        cloner=Cloner();paths=[f'/World/envs/env_{i}' for i in range(n)]
        positions=np.zeros((n,3),np.float32)
        if self.source_port:
            cols=int(np.ceil(np.sqrt(n)));positions[:,0]=(np.arange(n)%cols)*12.;positions[:,1]=(np.arange(n)//cols)*4.
        else:positions[:,1]=np.arange(n)*20.
        cloner.clone('/World/envs/env_0',paths,positions=positions,replicate_physics=False,copy_from_source=True)
        if self.source_port:
            scene_path=str(self.world.get_physics_context().get_current_physics_scene_prim().GetPath())
            cloner.filter_collisions(scene_path,'/World/collisionGroups',paths,[f'/World/floor_g{self.m.geom("floor").id}'])
        self.view=self.world.scene.add(Articulation('/World/envs/env_*/Bike',name='bikes'))
        self.world.reset();self.names=self.view.dof_names;self.pv=self.view._physics_view
        self.origins=np.array([positions[int(path.split('/env_')[1].split('/')[0])] for path in self.view.prim_paths],np.float32)
        self.qa=np.array([self.m.joint(name).qposadr[0] for name in self.names]);self.va=np.array([self.m.joint(name).dofadr[0] for name in self.names])
        self.ai=np.array([self.names.index(self.m.joint(j).name) for j in self.m.actuator_trnid[:,0]])
        self.arm=self.ai[2:4];self.baseq=self.m.key_qpos[0,self.qa].astype(np.float32)
        self.wheels=np.array([self.names.index(n) for n in ['frontwheel_joint','rearwheel_joint']]);self.previous_wheel_q=np.zeros((n,2));self.continuous_wheel_q=np.zeros((n,2))
        self.arm_kp=np.array([spec['target_drive_gains'][self.names[j]]['kp'] for j in self.arm])
        self.arm_kd=np.array([spec['target_drive_gains'][self.names[j]]['kd'] for j in self.arm])
        self.arm_cap=self.m.actuator_forcerange[2:4,1]
        kps,kds=self.view.get_gains();kps[:,self.arm]=0;kds[:,self.arm]=0;self.view.set_gains(kps=kps,kds=kds)
        self.effort=np.zeros((n,5),np.float32);self.observers=[None]*n;self.events=[None]*n
        self.last_action=np.zeros((n,4),np.float32);self.returns=np.zeros(n);self.initial_x=np.zeros(n);self.airborne=np.zeros(n,bool)
        self.obs=np.zeros((n,76),np.float32);self.privileged_obs=np.zeros((n,106),np.float32);self.prev_vel=np.zeros((n,3))
        self.source_resets=None
        if self.source_port:
            from source_reset import SourceResetStream,initial_env_keys
            self.source_resets=SourceResetStream(initial_env_keys(seed,n),spec['reset'],float(self.m.key_qpos[0,0]),float(self.m.key_qpos[0,2]))
        self.reset(np.arange(n))
        self.audit={'backend':'real PhysX CPU articulation dynamics','num_envs':n,'dof_names':self.names,'prim_paths':self.view.prim_paths,
                    'effective_pd':spec['target_drive_gains'],'native_gains':[x.tolist() for x in self.view.get_gains()],
                    'hip_knee_mode':'explicit capped PD, applied efforts saved and used for reward',
                    'wheel_static_dynamic_friction':[.5,.5],'all_material_combine':'min',
                    'source_anisotropic_contact_preserved':False,'physics_dt':self.dt,'control_dt':spec['control_dt'],
                    'body_names':self.view.body_names,'body_masses':self.pv.get_masses()[0].tolist(),
                    'mj_step_calls':0,'actor_dimensions':76,'conversion':conv,
                    'source_ppo_port':self.source_port,'critic_dimensions':106 if self.source_port else 76,
                    'enhanced_determinism':self.source_port,'origins':self.origins.tolist()}

    def hard_reset(self):
        """Restart the solver before scored panels, then restore hybrid actuators."""
        self.world.reset(soft=False);self.pv=self.view._physics_view
        kps,kds=self.view.get_gains();kps[:,self.arm]=0;kds[:,self.arm]=0
        self.view.set_gains(kps=kps,kds=kds);self.effort[:]=0
        self.reset(np.arange(self.n))

    def reset(self,indices,seeds=None,ground_only=False):
        ids=np.asarray(indices,dtype=np.int32);count=len(ids)
        if not count:return self.obs
        pos=np.tile(self.m.key_qpos[0,:3],(count,1));quat=np.tile(self.m.key_qpos[0,3:7],(count,1));vel=np.zeros((count,6),np.float32)
        vel[:,0]=CFG.reset.initial_forward_velocity
        for k,i in enumerate(ids):
            rng=np.random.default_rng(seeds[k]) if seeds is not None else self.rng[i]
            air=(not ground_only) and rng.random()<CFG.reset.airborne_rsi_probability;self.airborne[i]=air
            if air:
                for field,col in [('x',0),('z',2)]:pos[k,col]=rng.uniform(getattr(CFG.reset,'airborne_rsi_'+field+'_min'),getattr(CFG.reset,'airborne_rsi_'+field+'_max'))
                for field,col in [('vx',0),('vz',2)]:vel[k,col]=rng.uniform(getattr(CFG.reset,'airborne_rsi_'+field+'_min'),getattr(CFG.reset,'airborne_rsi_'+field+'_max'))
            if self.source_resets is not None and seeds is None and not ground_only:
                x,z,vx,vz,air=self.source_resets.pending[i];pos[k,0]=x;pos[k,2]=z;vel[k,0]=vx;vel[k,2]=vz;self.airborne[i]=bool(air)
            self.initial_x[i]=pos[k,0];self.events[i]=semantics.initial_event_state(np.asarray(pos[k,0]),CFG)
            self.observers[i]=Observation(self.m);self.obs[i]=self.observers[i].initial(pos[k,0]);self.returns[i]=0;self.last_action[i]=0;self.prev_vel[i]=vel[k,:3]
            if self.source_port:
                observer=self.observers[i];observer.d.qpos[:]=self.m.key_qpos[0];observer.d.qpos[:3]=pos[k];observer.d.qpos[3:7]=quat[k];mujoco.mj_kinematics(self.m,observer.d)
                self.privileged_obs[i]=observer.privileged(self.obs[i],vel[k,:3],np.zeros(3),np.zeros(5),self.names)
        self.view.set_world_poses((pos+self.origins[ids]).astype(np.float32),quat.astype(np.float32),indices=ids)
        self.view.set_velocities(vel,indices=ids)
        q=np.tile(self.baseq,(count,1));self.view.set_joint_positions(q,indices=ids);self.view.set_joint_velocities(np.zeros((count,5),np.float32),indices=ids)
        self.previous_wheel_q[ids]=q[:,self.wheels];self.continuous_wheel_q[ids]=q[:,self.wheels]
        self.view.set_joint_position_targets(q,indices=ids);self.view.set_joint_velocity_targets(np.zeros((count,5),np.float32),indices=ids)
        self.view.set_joint_efforts(np.zeros((count,5),np.float32),indices=ids)
        self.world.physics_sim_view.update_articulations_kinematic()
        return self.obs

    def step(self,actions):
        actions=np.asarray(actions,np.float32);assert actions.shape==(self.n,4) and np.isfinite(actions).all()
        if self.source_resets is not None:self.source_resets.advance()
        ctrl=np.stack([control(a,self.m,RAW) for a in actions]).astype(np.float32)
        targets=np.tile(self.baseq,(self.n,1));targets[:,self.ai[0]]=ctrl[:,0];targets[:,self.arm]=ctrl[:,2:4]
        speeds=np.zeros_like(targets);speeds[:,self.ai[1]]=ctrl[:,1]
        self.view.set_joint_position_targets(targets);self.view.set_joint_velocity_targets(speeds)
        for _ in range(self.substeps):
            q=self.view.get_joint_positions();qd=self.view.get_joint_velocities()
            self.effort[:,self.arm]=pd_effort(q[:,self.arm],qd[:,self.arm],targets[:,self.arm],self.arm_kp,self.arm_kd,self.arm_cap)
            self.view.set_joint_efforts(self.effort)
            self.world.step(render=False)
            pos,quat=self.view.get_world_poses();omega=self.view.get_angular_velocities()
            rotation=Rotation.from_quat(quat[:,[1,2,3,0]])
            velocity=self.view.get_linear_velocities()-np.cross(omega,rotation.apply(np.tile(self.m.body_ipos[1],(self.n,1))))
            acceleration=(velocity-self.prev_vel)/self.dt;self.prev_vel=velocity.copy()
        pos=pos-self.origins;q=self.view.get_joint_positions();qd=self.view.get_joint_velocities()
        if self.source_port:
            self.continuous_wheel_q=continuous_joint_position(self.previous_wheel_q,q[:,self.wheels],self.continuous_wheel_q);self.previous_wheel_q=q[:,self.wheels].copy()
        if not np.isfinite(np.concatenate([pos,quat,q,qd,velocity,omega],axis=1)).all():raise RuntimeError('Nonfinite PhysX state; stop rather than silently reset')
        angles=rotation.as_euler('xyz');rew=np.zeros(self.n,np.float32);done=np.zeros(self.n,bool);infos=[]
        actual=self.pv.get_dof_actuation_forces().copy()
        if not np.allclose(actual[:,self.arm],self.effort[:,self.arm],atol=1e-5):raise RuntimeError('Applied hip/knee effort readback mismatch')
        for i in range(self.n):
            e=self.events[i];roll,pitch,yaw=angles[i]
            illegal=False
            if self.source_port:
                self.obs[i]=self.observers[i].advance(pos[i],quat[i],q[i],qd[i],self.names,velocity[i],omega[i],acceleration[i],actions[i])
                self.privileged_obs[i]=self.observers[i].privileged(self.obs[i],velocity[i],omega[i],qd[i],self.names)
                self.privileged_obs[i,76+self.qa[self.wheels]]=self.continuous_wheel_q[i]
                illegal=bool(self.privileged_obs[i,-1])
            initial_x=float(self.m.key_qpos[0,0]) if self.source_port else self.initial_x[i]
            ti=semantics.TerminalInputs(e.episode_step,np.asarray(False),roll,pitch,np.asarray(illegal),np.asarray(pos[i,0]<initial_x-CFG.physical_limits.max_backward_distance),e.stuck,yaw,e.jump_zone_seen)
            pre=semantics.classify_terminal(ti,CFG)
            self.events[i]=semantics.advance_events(e,semantics.PhaseUSignals(pos[i,0],pos[i,2],velocity[i,0],velocity[i,2],pre.terminated),CFG)
            ev=self.events[i];terminal=semantics.classify_terminal(ti.replace(stuck=ev.stuck,jump_zone_seen=ev.jump_zone_seen),CFG)
            rs=rewards.RewardState(*pos[i],roll,pitch,yaw,*velocity[i],*omega[i],*qd[i,self.arm],*actual[i,self.arm])
            ri=rewards.RewardInputs(rs,actions[i],self.last_action[i],ev.jump_signal,ev.apex_seen & ~e.apex_seen,np.asarray(illegal),terminal.physical_failure,terminal.roll_limit|terminal.pitch_limit,terminal.jump_zone_missed,terminal.stuck,terminal.yaw_limit,terminal.timeout)
            rr=rewards.phase_u_reward(ri,CFG.reward,CFG.physical_limits);reward=float(rr.total)
            if terminal.stuck or terminal.yaw_limit:reward=CFG.reward.failed_episode_return-self.returns[i]
            self.returns[i]+=reward;rew[i]=reward;done[i]=terminal.terminated|terminal.truncated
            if not self.source_port:self.obs[i]=self.observers[i].advance(pos[i],quat[i],q[i],qd[i],self.names,velocity[i],omega[i],acceleration[i],actions[i])
            else:self.obs[i,-1]=ev.jump_signal;self.privileged_obs[i,:76]=self.obs[i]
            infos.append({'end_code':int(terminal.end_code),'terminated':bool(terminal.terminated),'truncated':bool(terminal.truncated),'apex':bool(ev.apex_seen),'return':float(self.returns[i]),'steps':int(ev.episode_step),'airborne_reset':bool(self.airborne[i]),'vx':float(velocity[i,0]),'x':float(pos[i,0]),'z':float(pos[i,2]),'roll':float(roll),'reward':float(reward),'reward_unadjusted':float(rr.total),'components':{k:float(v) for k,v in rr.components.as_dict().items()},'reward_state':list(map(float,[*pos[i],roll,pitch,yaw,*velocity[i],*omega[i],*qd[i,self.arm],*actual[i,self.arm]]))})
            infos[-1]['reward_flags']=[bool(x) for x in [ev.jump_signal,ev.apex_seen & ~e.apex_seen,illegal,terminal.physical_failure,terminal.roll_limit|terminal.pitch_limit,terminal.jump_zone_missed,terminal.stuck,terminal.yaw_limit,terminal.timeout]]
            infos[-1]['action']=actions[i].tolist();infos[-1]['last_action']=self.last_action[i].tolist();infos[-1]['return_before']=float(self.returns[i]-reward)
        self.last_action=actions.copy()
        self.trace=np.concatenate([np.zeros((self.n,1)),pos,quat,q,qd,velocity,omega,acceleration,ctrl],axis=1)
        return self.obs.copy(),rew,done,infos
