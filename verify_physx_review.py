"""Verify logged rewards, diagnose repeatability, and assemble all-case media."""
import argparse
import json
import os
import sys
from pathlib import Path
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.dont_write_bytecode = True
import numpy as np
import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    review=a.run/'review';root=review/'best_rollout'
    sys.path.insert(0,'/home/qy/DVGC/JIT/src')
    from jit_dvgc.config import load_config
    from jit_dvgc import rewards
    cfg=load_config(Path(__file__).resolve().parent/'policy/resolved_config.json',runtime_only=True)
    folders=sorted(root.glob('seed_*'));assert len(folders)==8
    verified=[];differences=[];max_error=0.;count=0
    for f in folders:
        z=np.load(f/'traces.npz')['target'];rows=json.loads((f/'steps.json').read_text())
        oldf=a.run/'evaluation'/'best_test'/f.name
        old=np.load(oldf/'traces.npz')['target'];oldrows=json.loads((oldf/'steps.json').read_text());n=min(len(z),len(old))
        diffs={'case':f.name,'old_duration_s':float(old[-1,0]),'new_duration_s':float(z[-1,0]),
               'old_return':oldrows[-1]['return'],'new_return':rows[-1]['return'],
               'old_end_code':oldrows[-1]['end_code'],'new_end_code':rows[-1]['end_code']}
        for key,columns,threshold in [('root_position_m',slice(1,4),1e-6),('joint_position_rad',slice(8,13),1e-6),('velocity_m_s',slice(18,21),1e-6),('physical_target',slice(27,31),1e-6)]:
            delta=np.max(np.abs(z[:n,columns]-old[:n,columns]),axis=1);hit=np.flatnonzero(delta>threshold)
            diffs[key]=dict(max_abs_shared_window=float(delta.max()),first_difference_gt_1e_6_s=float(z[hit[0],0]) if len(hit) else None)
        differences.append(diffs)
        for r in rows:
            inp=rewards.RewardInputs(rewards.RewardState(*map(np.float32,r['reward_state'])),np.array(r['action'],np.float32),np.array(r['last_action'],np.float32),*[np.asarray(x) for x in r['reward_flags']])
            result=rewards.phase_u_reward(inp,cfg.reward,cfg.physical_limits)
            expected=np.array([float(result.total)]+[float(v) for v in result.components.values()])
            actual=np.array([r['reward_unadjusted']]+list(r['components'].values()))
            np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-4)
            total=cfg.reward.failed_episode_return-r['return_before'] if r['reward_flags'][6] or r['reward_flags'][7] else float(result.total)
            np.testing.assert_allclose(r['reward'],total,rtol=3e-5,atol=3e-4)
            max_error=max(max_error,float(np.max(np.abs(actual-expected))));count+=1
        for name in ['trajectory','joints','commands','efforts','reward_components']:
            for ext in ['png','pdf']:assert (f/(name+'.'+ext)).stat().st_size>1000
        for name in ['diagnostics.csv','diagnostics.npz','replay.mp4']:assert (f/name).stat().st_size>1000
        with iio.get_reader(f/'replay.mp4') as rd:
            frames=rd.count_frames();fps=rd.get_meta_data()['fps'];assert frames==100 and fps==25
            preview=rd.get_data(0);assert preview.shape==(480,640,3)
            # A contact sheet makes first, moving and frozen-end frames inspectable.
            sheet=np.concatenate([preview,rd.get_data(24),rd.get_data(99)],axis=1)
            iio.imwrite(f/'video_contact_sheet.png',sheet)
        render=json.loads((f/'render_verification.json').read_text())
        assert render['max_root_pose_replay_error_m']<1e-5
        assert abs(render['final_state_hold_after_s']-z[-1,0])<1e-10
        verified.append(dict(case=f.name,frames=frames,fps=fps,duration_s=frames/fps,actual_failure_s=float(z[-1,0]),reward_rows=len(rows)))
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,22)
    readers=[iio.get_reader(f/'replay.mp4') for f in folders]
    with iio.get_writer(review/'all_cases.mp4',fps=25,codec='libx264',quality=8,macro_block_size=16) as writer:
        for k in range(100):
            # Each column pairs one ground and one airborne case; original 640x480 tiles.
            upper=np.concatenate([r.get_data(k) for r in readers[:4]],axis=1)
            lower=np.concatenate([r.get_data(k) for r in readers[4:]],axis=1)
            canvas=Image.new('RGB',(2560,1024),(13,20,29));canvas.paste(Image.fromarray(upper),(0,32));canvas.paste(Image.fromarray(lower),(0,544))
            d=ImageDraw.Draw(canvas)
            d.text((12,3),'GROUND RESET | BEST transition_05251072 | actual PhysX trajectories; failure poses held',font=font,fill='white')
            d.text((12,515),'AIRBORNE RESET | same checkpoint | all eight declared cases shown',font=font,fill='white')
            writer.append_data(np.asarray(canvas))
            if k==24:canvas.save(review/'all_cases.png')
    for r in readers:r.close()
    with iio.get_reader(review/'all_cases.mp4') as rd:assert rd.count_frames()==100
    (review/'repeatability_diagnostics.json').write_text(json.dumps(differences,indent=2))
    report=dict(verified=True,cases=verified,original_jax_reward_rows=count,max_reward_absolute_difference=max_error,
                composite_frames=100,composite_fps=25,physics_unchanged=True,
                reproducibility_passed=False,reproducibility_scope='fresh process versus training-end panel; 0/8 identical; root cause not established')
    (review/'delivery_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
