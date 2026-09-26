"""Evidence checks for the actuator audit; never equates these with sim2sim success."""
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from select_actuator_calibration import loss

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/actuator_report'
checks = {}
hashes = {}
source = Path('/home/qy/DVGC/assets/orange_bike_4kg_horizontal.xml')
checkpoint = Path('/home/qy/DVGC/JIT/runs/phase_u/phase_u_v4_speed2_roll400_missed200_9977856_seed820701_20260826/checkpoints/transition_4988928')
expected = [(source, '0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a'),
            (ROOT/'model/source.xml', '0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a'),
            (checkpoint/'payload.pkl', 'b388de198628b56309e83b3b4993ecf0c655f341c259883a2d43f19e954c6f5e')]
for path, value in expected:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == value, (path, digest)
    hashes[str(path)] = digest
checks['original_model_and_checkpoint_preserved'] = True
count = 0
for run in ['actuator_audit','actuator_calibration','actuator_calibration_validation']:
    folder = ROOT/'results'/run
    declaration = json.loads((folder/'declaration.json').read_text())
    status = json.loads((folder/'status.json').read_text())
    assert status['completed'] and status['cases']==len(declaration['cases'])
    assert status['training_interactions']==0
    for case in declaration['cases']:
        d = np.load(folder/case['name']/'traces.npz')
        n = len(d['joint_names']); samples=round(case['duration']/case['dt'])+1
        for engine in ['physx','RK4','IMPLICITFAST']:
            assert d[engine].shape==(samples,1+2*n)
            assert np.isfinite(d[engine]).all()
            np.testing.assert_allclose(d[engine][:,0],np.arange(samples)*case['dt'],atol=5e-8)
        np.testing.assert_allclose(d['physx'][0],d['RK4'][0],atol=1e-7)
        if case['mode']=='torque':
            np.testing.assert_array_equal(d['RK4_instantaneous_motor_torque'],0.)
        count += 1
checks['all_91_paired_cases_finite_time_and_initial_state_checked'] = count==91
assert count==91
declaration = json.loads((ROOT/'results/actuator_calibration/declaration.json').read_text())
reference = {}
for case in declaration['cases']:
    d=np.load(ROOT/'results/actuator_calibration'/case['name']/'traces.npz')
    if case['joint'] in reference: np.testing.assert_array_equal(d['RK4'],reference[case['joint']])
    else: reference[case['joint']] = d['RK4'].copy()
checks['target_parameter_search_did_not_change_source_trajectories'] = True
selection=json.loads((ROOT/'results/actuator_calibration/selection.json').read_text())
for joint, item in selection.items():
    scores=[loss(ROOT/'results/actuator_calibration'/r['case']['name'])['loss'] for r in item['candidates']]
    assert np.isclose(min(scores),item['best']['loss'])
    assert item['adopted'] is False
checks['frozen_selection_recomputed'] = True
summary=json.loads((OUT/'summary.json').read_text())
assert summary['passive_gate_passed'] and summary['fixed_rotor_gate_passed']
assert not summary['sim2sim_accepted'] and not summary['fitted_candidate_adopted']
for folder in ['actuator_mjx_rotor','actuator_mjx_bike']:
    rs=json.loads((ROOT/'results'/folder/'summary.json').read_text())
    for r in rs:
        name=r['case']['name']; ref=np.load(ROOT/'results/actuator_audit'/name/'traces.npz')['RK4']
        tr=np.load(ROOT/'results'/folder/name/'trace.npz')['mjx']; assert np.isfinite(tr).all()
        n=(tr.shape[1]-1)//2
        assert np.max(np.abs(tr[:,1+n:]-ref[:,1+n:]))<.0005
checks['four_native_mjx_probes_match_cpu_within_0_0005_rad_s'] = True
video=imageio.get_reader(OUT/'comparison.mp4'); frames=[f for f in video]; video.close()
assert len(frames)==75 and frames[0].shape==(480,1280,3)
assert np.mean(np.abs(frames[0].astype(float)-frames[20]))>.5
checks['video_75_nonempty_frames_and_changes'] = True
for case in ['baseline','calibrated_drives']:
    f=ROOT/'results/actuator_policy_smoke'/case
    r=json.loads((f/'result.json').read_text()); v=json.loads((f/'render_verification.json').read_text())
    tr=np.load(f/'traces.npz')['target']
    assert r['target_first_failure_s']==r['duration']
    assert r['target_endpoint']['terminated'] and not r['target_screen_qualified']
    assert np.isclose(tr[-1,0],r['duration']) and v['physics_steps_during_render']==0
checks['policy_failures_and_replay_not_misrepresented'] = True
for script in ['actuator_audit.py','actuator_reference.py','actuator_mjx_check.py','drive_calibration.py','select_actuator_calibration.py','diagnose_domains.py','report_actuator_audit.py']:
    hashes[script]=hashlib.sha256((ROOT/script).read_bytes()).hexdigest()
checks['all_checks_passed']=all(checks.values())
(OUT/'verification.json').write_text(json.dumps(dict(checks=checks,sha256=hashes,sim2sim_accepted=False),indent=2))
print(json.dumps(checks,indent=2))
