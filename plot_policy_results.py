from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parent;p=root/'results/transition_4988928';z=np.load(p/'trajectory.npz');ref=np.load(root/'results/mujoco_checkpoint_reference/trajectory.npz');pol=np.load(p/'policy_trace.npz')
fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained');summary={}
for data,label,color in [(z,'Isaac / PhysX','#d1495b'),(ref,'MuJoCo CPU reference','#00798c')]:
 t=data['time'];pos=data['base_position'];e=Rotation.from_quat(data['base_quaternion'][:,[1,2,3,0]]).as_euler('xyz');roll=np.degrees(e[:,0]);ids=np.flatnonzero(abs(roll)>35)
 axs[0,0].plot(pos[:,0],pos[:,1],label=label,color=color);axs[0,1].plot(t,pos[:,2],label=label,color=color);axs[0,2].plot(t,roll,label=label,color=color)
 summary[label]={'max_root_height_m':float(pos[:,2].max()),'max_root_x_m':float(pos[:,0].max()),'first_roll_over_35deg_s':float(t[ids[0]]) if len(ids) else None,'first_jump_zone_entry_s':float(t[np.flatnonzero((pos[:,0]>=2.5)&(pos[:,0]<=4))[0]]) if np.any((pos[:,0]>=2.5)&(pos[:,0]<=4)) else None}
 for a in axs[0,:]:a.legend()
axs[0,0].set(xlabel='World x (m)',ylabel='World y (m)',title='Actual XY (full diagnostic horizon)');axs[0,1].set(xlabel='Time (s)',ylabel='Root height (m)',title='Root height');axs[0,1].axhline(.5,color='gray',ls='--');axs[0,2].set(xlabel='Time (s)',ylabel='Roll (deg)',title='Roll; source limit +/-35 deg');axs[0,2].axhline(35,color='gray',ls='--');axs[0,2].axhline(-35,color='gray',ls='--')
for i,n in enumerate(['steer','drive','hip','knee']):axs[1,0].plot(pol['time'],pol['actions'][:,i],label=n)
axs[1,0].legend();axs[1,0].set(xlabel='Time (s)',ylabel='Normalized action',title='Frozen Actor output in Isaac')
for i,n in enumerate(['steer','hip','knee']):j=json.loads((p/'result.json').read_text())['joint_names'].index({'steer':'steering_joint','hip':'hip_joint','knee':'knee_joint'}[n]);axs[1,1].plot(z['time'],z['q'][:,j],label=n)
axs[1,1].legend();axs[1,1].set(xlabel='Time (s)',ylabel='Joint position (rad)',title='Isaac physical joint states')
axs[1,2].plot(pol['time'],pol['observations'][:,-1]);axs[1,2].set(xlabel='Time (s)',ylabel='Jump signal',title='Isaac observed jump trigger',ylim=(-.1,1.1))
for a in axs.flat:a.grid(alpha=.25)
fig.suptitle('Frozen transition_4988928 | 8s engineering diagnostic | no retraining\nGround/tire visuals changed only; contact/servo differences remain')
fig.savefig(p/'comparison.png',dpi=160);fig.savefig(p/'comparison.pdf');(p/'comparison_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
np.savetxt(p/'policy_actions.csv',np.column_stack([pol['time'],pol['actions'],pol['controls']]),delimiter=',',header='time_s,action_steer,action_drive,action_hip,action_knee,target_steer_rad,target_drive_rad_s,target_hip_rad,target_knee_rad',comments='')
