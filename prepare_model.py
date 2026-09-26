"""Read-only MuJoCo source audit; produce an explicit-inertia importer input."""
import argparse, hashlib, json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,default=Path('/home/qy/DVGC/assets/orange_bike_4kg_horizontal.xml'))
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
m=mujoco.MjModel.from_xml_path(str(a.source))
compiled=a.output/'compiled.xml'; mujoco.mj_saveLastXML(str(compiled),m)
tree=ET.parse(compiled); compiler=tree.getroot().find('compiler')
compiler.set('meshdir',str(a.source.parent/'meshes')); compiler.set('inertiafromgeom','false')
for body in tree.getroot().iter('body'):
    i=m.body(body.attrib['name']).id
    old=body.find('inertial')
    if old is not None: body.remove(old)
    ET.SubElement(body,'inertial',{
        'mass':repr(float(m.body_mass[i])),
        'pos':' '.join(map(str,m.body_ipos[i])),
        'quat':' '.join(map(str,m.body_iquat[i])),
        'diaginertia':' '.join(map(str,m.body_inertia[i]))})
tree.write(compiled,encoding='unicode')
n=mujoco.MjModel.from_xml_path(str(compiled))
for name in ['body_mass','body_ipos','body_inertia','body_iquat','jnt_pos','jnt_axis','actuator_gainprm','actuator_biasprm']:
    np.testing.assert_allclose(getattr(m,name),getattr(n,name),rtol=1e-5,atol=1e-7)
def name(obj,i):return mujoco.mj_id2name(m,obj,i)
audit={'source':str(a.source),'sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
       'mujoco_version':mujoco.__version__,'total_mass_kg':float(m.body_mass.sum()),
       'xml_physics_dt_s':m.opt.timestep,'jit_physics_dt_s':.005,'jit_control_dt_s':.020,
       'bodies':[{'name':name(mujoco.mjtObj.mjOBJ_BODY,i),'mass':float(m.body_mass[i]),'com':m.body_ipos[i].tolist(),'principal_inertia':m.body_inertia[i].tolist(),'inertia_quat_wxyz':m.body_iquat[i].tolist()} for i in range(1,m.nbody)],
       'joints':[{'name':name(mujoco.mjtObj.mjOBJ_JOINT,i),'type':int(m.jnt_type[i]),'axis':m.jnt_axis[i].tolist(),'range':m.jnt_range[i].tolist()} for i in range(m.njnt)],
       'actuators':[{'name':name(mujoco.mjtObj.mjOBJ_ACTUATOR,i),'joint':name(mujoco.mjtObj.mjOBJ_JOINT,int(m.actuator_trnid[i,0])),'gain':m.actuator_gainprm[i].tolist(),'bias':m.actuator_biasprm[i].tolist(),'ctrlrange':m.actuator_ctrlrange[i].tolist(),'forcerange':m.actuator_forcerange[i].tolist()} for i in range(m.nu)],
       'compiled_equivalence_check':'passed; mass, COM, inertia, axes, anchors and actuator coefficients',
       'limitations':['PhysX contact solver differs from MuJoCo','condim6 anisotropy and solref/solimp require a separately declared contact approximation','asset smoke does not transfer a learned controller']}
(a.output/'source_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
