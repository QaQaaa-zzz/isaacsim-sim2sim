from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import imageio.v2 as imageio
root=Path(__file__).resolve().parent
for mode in ['fixed','free']:
 p=root/'results'/f'final_{mode}';z=np.load(p/'trajectory.npz');r=json.loads((p/'result.json').read_text());t=z['time'];q=z['base_quaternion'];e=Rotation.from_quat(q[:,[1,2,3,0]]).as_euler('xyz')
 fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
 for i,n in enumerate(r['joint_names']):ax[0,0].plot(t,np.unwrap(z['q'][:,i]),label=n)
 ax[0,0].set(ylabel='Joint angle (rad)',title='Angles (continuous joints unwrapped)');ax[0,0].legend(fontsize=7)
 for i,n in enumerate(r['joint_names']):ax[0,1].plot(t,z['qd'][:,i],label=n)
 ax[0,1].set(ylabel='Joint velocity (rad/s)',title='Physical joint velocities')
 for i,n in enumerate(['x','y','z']):ax[1,0].plot(t,z['base_position'][:,i],label=n)
 ax[1,0].set(ylabel='Base position (m)');ax[1,0].legend()
 for i,n in enumerate(['roll','pitch','yaw']):ax[1,1].plot(t,np.degrees(e[:,i]),label=n)
 ax[1,1].set(ylabel='Base attitude (deg)');ax[1,1].legend()
 for a in ax.flat:a.set_xlabel('Time (s)');a.grid(alpha=.25)
 fig.suptitle(f'DVGC / Isaac Sim 5.1 / {mode} base / no learned balance controller')
 fig.savefig(p/'states.png',dpi=160);fig.savefig(p/'states.pdf');plt.close(fig)
 reader=imageio.get_reader(p/'simulation.mp4');imageio.imwrite(p/'last.png',reader.get_data(reader.count_frames()-1));reader.close()
 summary={'mode':mode,'steps':len(t),'duration_s':float(t[-1]),'frames':r['frames'],'mass_kg':5.091,'max_abs_roll_deg':float(np.max(np.abs(np.degrees(e[:,0])))),'final_roll_deg':float(np.degrees(e[-1,0])),'first_abs_roll_over_45deg_s':float(t[np.flatnonzero(np.abs(e[:,0])>np.pi/4)[0]]) if np.any(np.abs(e[:,0])>np.pi/4) else None,'note':'45deg is a descriptive fall marker, not a predeclared task-success oracle','max_fk_position_error_m':r['max_fk_position_error_m'],'max_render_body_position_error_m':r['max_render_body_position_error_m']}
 (p/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)
