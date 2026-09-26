"""Artifact consistency checks, explicitly separate from sim2sim acceptance."""
from pathlib import Path
import json,hashlib
import numpy as np,imageio.v2 as iio
from policy_runtime import Actor
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
identity=json.loads((ROOT/'policy/identity.json').read_text());source=Path('/home/qy/DVGC/assets/orange_bike_4kg_horizontal.xml');checkpoint=Path('/home/qy/DVGC/JIT/runs/phase_u/phase_u_v4_speed2_roll400_missed200_9977856_seed820701_20260826/checkpoints/transition_4988928/payload.pkl')
assert sha(source)==sha(ROOT/'model/source.xml')==identity['xml_sha256'];assert sha(checkpoint)==identity['payload_sha256']
ref=np.load(ROOT/'policy/reference_inference.npz');error=float(np.max(np.abs(Actor(ROOT/'policy/actor.npz')(ref['observations'])-ref['actions'])));assert error<3e-5
r=json.loads((ROOT/'results/final_target_test/summary.json').read_text());assert len(r)==36
for row in r:
 d=ROOT/'results/final_target_test'/row['case'];z=np.load(d/'traces.npz')['target'];assert np.isfinite(z).all();assert np.all(np.diff(z[:,0])>0);assert np.isclose(row['duration'],z[-1,0]);assert len(z)*.02<=3.000001
 if row['target_first_failure_s']:assert row['target_first_failure_s']==row['duration']
count={who:sum(x['target_screen_qualified'] for x in r if x['case'].startswith('test_') and x['case'].endswith('_'+who)) for who in ['original','dr']};assert count=={'original':0,'dr':0}
videos={}
for name in ['comparison']:
 p=ROOT/f'results/sim2sim_media/{name}.mp4';reader=iio.get_reader(p);n=reader.count_frames();assert n==75;first=reader.get_data(0);last=reader.get_data(n-1);assert first.shape==(960,1280,3);assert np.std(first)>8;videos[str(p.relative_to(ROOT))]={'frames':n,'first_last_mean_pixel_difference':float(np.mean(np.abs(first.astype(float)-last.astype(float))))};reader.close()
for name in ['final_original_live','final_dr_live']:
 row=json.loads((ROOT/f'results/{name}/result.json').read_text());assert row['terminal_status']['terminated'];assert row['max_render_body_position_error_m']<1e-6;assert row['max_fk_position_error_m']<1e-6
 if 'max_fk_orientation_error_rad' in row:assert row['max_fk_orientation_error_rad']<1e-5
for name,budget in [('dr_training',65536),('dr_solver_training',262144)]:
 status=json.loads((ROOT/f'results/{name}/status.json').read_text());assert status['completed'] and status['transitions']==budget
 for ck in (ROOT/f'results/{name}').glob('update_*'):
  ident=json.loads((ck/'identity.json').read_text());assert ident['actor_npz_sha256']==sha(ck/'actor.npz');assert not ident['adopted']
summary=json.loads((ROOT/'results/sim2sim_report/summary.json').read_text());assert summary['sim2sim_accepted'] is False and summary['policy_adopted'] is False
report={'artifact_checks_passed':True,'sim2sim_accepted':False,'target_qualified':count,'target_cases':36,'source_xml_sha256':sha(source),'source_checkpoint_sha256':sha(checkpoint),'original_actor_max_abs_error':error,'videos':videos,'training_transitions':327680}
(ROOT/'results/sim2sim_report/verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
