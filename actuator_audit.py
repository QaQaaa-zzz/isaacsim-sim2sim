"""Matched contact-free dynamics probes. No actor, observations, or training.

One config defines all cases. The complete original model is used for suspended
probes; a declared fixed-axis fixture isolates the rear rotor's exact inertia.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import mujoco
from actuator_reference import velocity_step

ROOT = Path(__file__).resolve().parent


def rotor_fixture(source, path):
    """Extract physical rotor parameters; diagnostic support is fixed to world."""
    m = mujoco.MjModel.from_xml_path(str(source))
    j = m.joint('rearwheel_joint').id
    b = m.jnt_bodyid[j]
    act = int(np.flatnonzero(m.actuator_trnid[:, 0] == j)[0])
    assert np.allclose(m.body_ipos[b], 0)
    assert np.allclose(m.body_iquat[b], [1, 0, 0, 0])
    assert np.allclose(m.jnt_axis[j], [0, 0, 1])
    inertia = m.body_inertia[b]
    kd = -float(m.actuator_biasprm[act, 2])
    limit = float(m.actuator_forcerange[act, 1])
    path.write_text(f'''<mujoco model="isolated_original_rear_rotor">
  <option gravity="0 0 0" timestep="0.001" integrator="RK4"/>
  <worldbody><body name="support">
    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
    <body name="rotor"><joint name="rearwheel_joint" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="{m.body_mass[b]:.17g}" diaginertia="{' '.join(format(x,'.17g') for x in inertia)}"/>
      <geom name="rotor_visual" type="cylinder" size=".07 .005" contype="0" conaffinity="0"/>
    </body></body></worldbody>
  <actuator><velocity name="rear_servo" joint="rearwheel_joint" kv="{kd}" forcerange="{-limit} {limit}"/></actuator>
  <keyframe><key qpos="0"/></keyframe>
</mujoco>''')
    return {'axial_inertia_kg_m2': float(inertia[2]), 'damping_Nm_s_rad': kd,
            'limit_Nm': limit, 'support': 'fixed world; no free-base recoil'}


def prepare(m, case, names):
    m.opt.gravity[:] = 0
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    m.opt.timestep = case['dt']
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    if case['model'] == 'bike':
        d.qpos[m.joint('hip_joint').qposadr] = -.8
        d.qpos[m.joint('knee_joint').qposadr] = 1.
    d.qvel[:] = 0
    for name, velocity in case.get('initial_joint_velocities', {}).items():
        d.qvel[m.joint(name).dofadr] = velocity
    qa = np.array([int(m.joint(n).qposadr[0]) for n in names])
    va = np.array([int(m.joint(n).dofadr[0]) for n in names])
    ai = np.array([names.index(m.joint(j).name) for j in m.actuator_trnid[:, 0]])
    if case['mode'] == 'torque':
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        d.qfrc_applied[va[names.index(case['joint'])]] = case['input']
    else:
        active = int(np.flatnonzero(ai == names.index(case['joint']))[0])
        for k in range(m.nu):
            if k != active:
                m.actuator_gainprm[k] = 0
                m.actuator_biasprm[k] = 0
        kp = -m.actuator_biasprm[active, 1]
        d.ctrl[active] = case['input'] + (d.qpos[qa[ai[active]]] if kp else 0.)
    mujoco.mj_forward(m, d)
    return d, qa, va, ai


def run_source(m, case, names, integrator):
    m = copy.copy(m)
    m.opt.integrator = getattr(mujoco.mjtIntegrator, 'mjINT_' + integrator)
    d, qa, va, ai = prepare(m, case, names)
    rows, motor = [], []
    for step in range(round(case['duration']/case['dt'])+1):
        # mj_step(RK4) leaves stage-dependent derived arrays; explicitly refresh.
        mujoco.mj_forward(m, d)
        rows.append(np.r_[d.time, d.qpos[qa], d.qvel[va]])
        motor.append(d.qfrc_actuator[va].copy())
        if step < round(case['duration']/case['dt']):
            mujoco.mj_step(m, d)
    return np.asarray(rows), np.asarray(motor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'declaration.json').write_text(json.dumps(spec, indent=2))
    source = ROOT/'model/source.xml'
    fixture = args.output/'rotor.xml'
    physical = rotor_fixture(source, fixture)
    (args.output/'fixture.json').write_text(json.dumps(physical, indent=2))
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'limit_cpu_threads': 4,
                         'extra_args': ['--/app/settings/loadUserConfig=false', '--/app/settings/persistent=false']})
    import omni.usd
    from pxr import UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from usd_model import build
    results = []
    for case in spec['cases']:
        started = time.monotonic()
        assert 0 < case['duration'] <= .1
        assert np.isclose(round(case['duration']/case['dt'])*case['dt'], case['duration'])
        out = args.output/case['name']; out.mkdir()
        (out/'case.json').write_text(json.dumps(case, indent=2))
        omni.usd.get_context().new_stage(); World.clear_instance()
        world = World(physics_dt=case['dt'], rendering_dt=.02)
        world.get_physics_context().enable_fabric(False)
        world.get_physics_context().set_gravity(0.)
        stage = omni.usd.get_context().get_stage()
        m, conversion = build(stage, fixture if case['model'] == 'rotor' else source,
                              fixed=case['model'] == 'rotor')
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        robot = world.scene.add(SingleArticulation('/World/Bike', name='probe'))
        world.reset(); names = robot.dof_names; n = len(names)
        original = copy.copy(m)
        d, qa, va, ai = prepare(m, case, names)
        robot.set_joint_positions(d.qpos[qa].astype(np.float32))
        robot.set_joint_velocities(d.qvel[va].astype(np.float32))
        if case['model'] == 'bike':
            robot.set_world_pose(d.qpos[:3], d.qpos[3:7])
            robot.set_linear_velocity(np.zeros(3)); robot.set_angular_velocity(np.zeros(3))
        view = robot._articulation_view
        kp = np.zeros(n); kd = np.zeros(n); tau = np.zeros(n)
        qt = d.qpos[qa].copy(); vt = np.zeros(n)
        joint_index = names.index(case['joint'])
        if case['mode'] == 'torque':
            tau[joint_index] = case['input']
        else:
            act = int(np.flatnonzero(ai == joint_index)[0])
            kp[joint_index] = -m.actuator_biasprm[act, 1]
            kd[joint_index] = -m.actuator_biasprm[act, 2]
            if kp[joint_index]: qt[joint_index] = d.ctrl[act]
            else: vt[joint_index] = d.ctrl[act]
        scales = case.get('target_drive_scales', {'kp': 1., 'kd': 1.})
        kp *= scales['kp']; kd *= scales['kd']
        view.set_gains(kps=kp[None].astype(np.float32), kds=kd[None].astype(np.float32))
        robot.apply_action(ArticulationAction(joint_positions=qt, joint_velocities=vt))
        robot.set_joint_efforts(tau.astype(np.float32))
        rows = []; base_rows = []
        for step in range(round(case['duration']/case['dt'])+1):
            q = robot.get_joint_positions(); qd = robot.get_joint_velocities()
            rows.append(np.r_[step*case['dt'], q, qd])
            pos, quat = robot.get_world_pose()
            base_rows.append(np.r_[pos, quat, robot.get_linear_velocity(), robot.get_angular_velocity()])
            if step < round(case['duration']/case['dt']):
                robot.set_joint_efforts(tau.astype(np.float32))
                world.step(render=False)
        target = np.asarray(rows)
        arrays = {'physx': target, 'physx_base': np.asarray(base_rows), 'joint_names': names}
        metrics = {'case': case, 'joint_names': names,
                   'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                   'runtime_gains': [v.tolist() for v in view.get_gains()],
                   'runtime_max_forces': view.get_max_efforts().tolist(),
                   'drive_force_note': 'Only isolated rotor: I*delta(v)/dt is actual net step-average joint torque. Full-body target drive force not separately measured.'}
        for integrator in spec['source_integrators']:
            tr, force = run_source(original, case, names, integrator)
            arrays[integrator] = tr; arrays[integrator+'_instantaneous_motor_torque'] = force
            error = target[:, 1+n:]-tr[:, 1+n:]
            metrics[integrator] = {
                'qd_rmse_rad_s': np.sqrt(np.mean(error**2, axis=0)).tolist(),
                'qd_relative_l2': float(np.linalg.norm(error)/max(np.linalg.norm(tr[:, 1+n:]), 1e-12)),
                'source_final_qd': tr[-1, 1+n:].tolist(),
                'physx_final_qd': target[-1, 1+n:].tolist()}
        if case['model'] == 'rotor':
            inertia = physical['axial_inertia_kg_m2']; ts = target[:, 0]
            if case['mode'] == 'servo':
                q, v = velocity_step(ts, inertia=inertia, damping=physical['damping_Nm_s_rad'],
                                     limit=physical['limit_Nm'], target=case['input'])
            else:
                q = .5*case['input']/inertia*ts**2; v = case['input']/inertia*ts
            arrays['analytic'] = np.column_stack([ts, q, v])
            for engine in ['physx'] + spec['source_integrators']:
                arrays[engine+'_step_average_torque'] = inertia*np.diff(arrays[engine][:, 2])/case['dt']
                metrics[engine+'_analytic'] = {
                    'velocity_rmse_rad_s': float(np.sqrt(np.mean((arrays[engine][:, 2]-v)**2))),
                    'position_rmse_rad': float(np.sqrt(np.mean((arrays[engine][:, 1]-q)**2))),
                    'velocity_relative_l2': float(np.linalg.norm(arrays[engine][:, 2]-v)/max(np.linalg.norm(v), 1e-12)),
                    'step_average_torque_peak_Nm': float(np.max(np.abs(arrays[engine+'_step_average_torque'])))}
        assert all(np.isfinite(value).all() for key, value in arrays.items() if key != 'joint_names')
        metrics['wall_s'] = time.monotonic()-started
        np.savez_compressed(out/'traces.npz', **arrays)
        (out/'result.json').write_text(json.dumps(metrics, indent=2))
        results.append(metrics)
        (args.output/'summary.json').write_text(json.dumps(results, indent=2))
        print('CASE_DONE', case['name'], metrics['RK4']['qd_relative_l2'], flush=True)
        world.stop()
    (args.output/'status.json').write_text(json.dumps({'completed': True, 'cases': len(results), 'training_interactions': 0}, indent=2))
    app.close()


if __name__ == '__main__':
    main()
