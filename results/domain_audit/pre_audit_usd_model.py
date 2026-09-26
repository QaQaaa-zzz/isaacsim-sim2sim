"""Explicit compiled-MuJoCo to USD mapping; requires initialized Isaac Sim.
Avoids MJCF importer ellipsoid fallback and absolute mesh-path crash.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from pxr import UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf, Sdf

def quat(q):return Gf.Quatf(float(q[0]),Gf.Vec3f(*map(float,q[1:])))
def rot(q):return Rotation.from_quat(np.r_[q[1:],q[0]])
def wxyz(r):
 q=r.as_quat();return np.r_[q[3],q[:3]]
def transform(prim,pos,q,scale=None):
 x=UsdGeom.Xformable(prim);x.AddTranslateOp().Set(Gf.Vec3d(*map(float,pos)));x.AddOrientOp().Set(quat(q))
 if scale is not None:x.AddScaleOp().Set(Gf.Vec3f(*map(float,scale)))
def mesh(stage,path,verts,faces):
 obj=UsdGeom.Mesh.Define(stage,path);obj.CreatePointsAttr(verts.tolist());obj.CreateFaceVertexCountsAttr([3]*len(faces));obj.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist());obj.CreateSubdivisionSchemeAttr('none');return obj

def build(stage,source,root_path='/World/Bike',fixed=False):
 m=mujoco.MjModel.from_xml_path(str(source));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
 UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
 robot=UsdGeom.Xform.Define(stage,root_path);UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())
 art=PhysxSchema.PhysxArticulationAPI.Apply(robot.GetPrim());art.CreateEnabledSelfCollisionsAttr(True);art.CreateSolverPositionIterationCountAttr(16);art.CreateSolverVelocityIterationCountAttr(4)
 bodypaths={0:None};geompaths={};colliders=[];audit=[]
 for b in range(1,m.nbody):
  name=m.body(b).name;path=f'{root_path}/{name}';bodypaths[b]=path;body=UsdGeom.Xform.Define(stage,path);prim=body.GetPrim();transform(prim,d.xpos[b],d.xquat[b])
  UsdPhysics.RigidBodyAPI.Apply(prim);mass=UsdPhysics.MassAPI.Apply(prim);mass.CreateMassAttr(float(m.body_mass[b]));mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float,m.body_ipos[b])));mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float,m.body_inertia[b])));mass.CreatePrincipalAxesAttr(quat(m.body_iquat[b]))
  rigid=PhysxSchema.PhysxRigidBodyAPI.Apply(prim);rigid.CreateLinearDampingAttr(0);rigid.CreateAngularDampingAttr(0)
  audit.append({'body':name,'source_mass':float(m.body_mass[b]),'usd_mass':mass.GetMassAttr().Get(),'source_inertia':m.body_inertia[b].tolist(),'usd_inertia':list(mass.GetDiagonalInertiaAttr().Get()),'source_com':m.body_ipos[b].tolist(),'usd_com':list(mass.GetCenterOfMassAttr().Get())})
 for g in range(m.ngeom):
  b=int(m.geom_bodyid[g]);name=m.geom(g).name or f'geom_{g}';path=f'{bodypaths[b] or "/World"}/{name}_g{g}';geompaths[g]=path
  typ=int(m.geom_type[g]);size=m.geom_size[g];scale=None
  if typ==mujoco.mjtGeom.mjGEOM_MESH:
   mid=int(m.geom_dataid[g]);va=int(m.mesh_vertadr[mid]);fa=int(m.mesh_faceadr[mid]);verts=m.mesh_vert[va:va+m.mesh_vertnum[mid]];faces=m.mesh_face[fa:fa+m.mesh_facenum[mid]];obj=mesh(stage,path,verts,faces)
  elif typ==mujoco.mjtGeom.mjGEOM_ELLIPSOID:
   # Surface triangles; convex hull cooking preserves axes with <0.6% chord error.
   verts=[];faces=[];nt=32;npole=16
   for j in range(npole+1):
    theta=np.pi*j/npole
    for k in range(nt):
     phi=2*np.pi*k/nt;verts.append(size*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]))
   for j in range(npole):
    for k in range(nt):
     a=j*nt+k;bb=j*nt+(k+1)%nt;c=a+nt;dd=bb+nt
     if j>0:faces.append([a,bb,c])
     if j<npole-1:faces.append([bb,dd,c])
   obj=mesh(stage,path,np.array(verts),np.array(faces));UsdPhysics.MeshCollisionAPI.Apply(obj.GetPrim()).CreateApproximationAttr('convexHull')
  elif typ==mujoco.mjtGeom.mjGEOM_BOX:
   obj=UsdGeom.Cube.Define(stage,path);obj.CreateSizeAttr(2);scale=size
  elif typ==mujoco.mjtGeom.mjGEOM_CYLINDER:
   obj=UsdGeom.Cylinder.Define(stage,path);obj.CreateRadiusAttr(float(size[0]));obj.CreateHeightAttr(float(2*size[1]));obj.CreateAxisAttr('Z')
  elif typ==mujoco.mjtGeom.mjGEOM_SPHERE:
   obj=UsdGeom.Sphere.Define(stage,path);obj.CreateRadiusAttr(float(size[0]))
  elif typ==mujoco.mjtGeom.mjGEOM_PLANE:
   # Physics infinite plane; large matching visual rectangle in its local XY plane.
   obj=UsdGeom.Plane.Define(stage,path);obj.CreateAxisAttr('Z');obj.CreateWidthAttr(200);obj.CreateLengthAttr(200)
  else:raise NotImplementedError(f'geometry type {typ}')
  prim=obj.GetPrim();transform(prim,m.geom_pos[g],m.geom_quat[g],scale)
  rgb=m.geom_rgba[g,:3].copy()
  if b==0:rgb=np.array([.19,.23,.29]) if typ==mujoco.mjtGeom.mjGEOM_PLANE else np.array([.4,.32,.21])
  obj.CreateDisplayColorAttr([Gf.Vec3f(*map(float,rgb))])
  collision=bool(m.geom_contype[g] or m.geom_conaffinity[g])
  if collision:
   UsdPhysics.CollisionAPI.Apply(prim);pc=PhysxSchema.PhysxCollisionAPI.Apply(prim);pc.CreateContactOffsetAttr(.001);pc.CreateRestOffsetAttr(0)
   if b>0:UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
   friction=float(m.geom_friction[g,0]);explicit_pair_ids=[i for i in range(m.npair) if g in (int(m.pair_geom1[i]),int(m.pair_geom2[i]))]
   if 'wheel' in name and explicit_pair_ids:friction=float(max(m.pair_friction[i,0] for i in explicit_pair_ids))
   mat=UsdShade.Material.Define(stage,f'/World/Materials/material_{g}');pm=UsdPhysics.MaterialAPI.Apply(mat.GetPrim());pm.CreateStaticFrictionAttr(friction);pm.CreateDynamicFrictionAttr(friction);pm.CreateRestitutionAttr(0)
   phmat=PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim());phmat.CreateFrictionCombineModeAttr('max');phmat.CreateRestitutionCombineModeAttr('min')
   UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics');colliders.append(g)
  elif float(m.geom_rgba[g,3])==0:UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
 # Explicitly preserve source collision masks/exclusions at geom pair level.
 explicit={tuple(sorted((int(m.pair_geom1[i]),int(m.pair_geom2[i])))) for i in range(m.npair)}
 excluded={int(v) for v in m.exclude_signature};filtered=0;allowed=[]
 for ix,g in enumerate(colliders):
  targets=[]
  for h in colliders[ix+1:]:
   bg,bh=int(m.geom_bodyid[g]),int(m.geom_bodyid[h]);pair=tuple(sorted((g,h)));sig=(min(bg,bh)<<16)+max(bg,bh)
   if pair in explicit:permit=True
   elif bg==bh or sig in excluded:permit=False
   elif bg and bh and (int(m.body_parentid[bg])==bh or int(m.body_parentid[bh])==bg):permit=False
   else:permit=bool((int(m.geom_contype[g])&int(m.geom_conaffinity[h])) or (int(m.geom_contype[h])&int(m.geom_conaffinity[g])))
   if not permit:targets.append(Sdf.Path(geompaths[h]));filtered+=1
   else:allowed.append([m.geom(g).name,m.geom(h).name])
  if targets:UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(geompaths[g])).CreateFilteredPairsRel().SetTargets(targets)
 for j in range(m.njnt):
  if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:continue
  assert m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE
  b=int(m.jnt_bodyid[j]);parent=int(m.body_parentid[b]);joint=UsdPhysics.RevoluteJoint.Define(stage,f'{root_path}/joints/{m.joint(j).name}');joint.CreateBody0Rel().SetTargets([Sdf.Path(bodypaths[parent])]);joint.CreateBody1Rel().SetTargets([Sdf.Path(bodypaths[b])]);joint.CreateAxisAttr('X');joint.CreateCollisionEnabledAttr(True)
  axis=d.xaxis[j];frame=Rotation.align_vectors(np.array([axis]),np.array([[1.,0,0]]))[0]
  for k,bid in [(0,parent),(1,b)]:
   rb=rot(d.xquat[bid]);pos=rb.inv().apply(d.xanchor[j]-d.xpos[bid]);q=wxyz(rb.inv()*frame)
   getattr(joint,f'CreateLocalPos{k}Attr')(Gf.Vec3f(*map(float,pos)));getattr(joint,f'CreateLocalRot{k}Attr')(quat(q))
  if m.jnt_limited[j]:joint.CreateLowerLimitAttr(float(np.degrees(m.jnt_range[j,0])));joint.CreateUpperLimitAttr(float(np.degrees(m.jnt_range[j,1])))
  pj=PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim());pj.CreateJointFrictionAttr(0)
 # USD angular gains use torque/degree; source gains use torque/radian.
 for act in range(m.nu):
  j=int(m.actuator_trnid[act,0]);name=m.joint(j).name;prim=stage.GetPrimAtPath(f'{root_path}/joints/{name}')
  drive=UsdPhysics.DriveAPI.Apply(prim,'angular');drive.CreateTypeAttr('force')
  kp=-float(m.actuator_biasprm[act,1]);kd=-float(m.actuator_biasprm[act,2]);gain=float(m.actuator_gainprm[act,0])
  drive.CreateStiffnessAttr(kp*np.pi/180);drive.CreateDampingAttr(kd*np.pi/180)
  assert np.isclose(m.actuator_forcerange[act,0],-m.actuator_forcerange[act,1])
  drive.CreateMaxForceAttr(float(m.actuator_forcerange[act,1]))
  drive.CreateTargetPositionAttr(float(np.degrees(m.key_qpos[0,m.jnt_qposadr[j]])) if kp else 0)
  drive.CreateTargetVelocityAttr(0)
 if fixed:
  joint=UsdPhysics.FixedJoint.Define(stage,f'{root_path}/joints/fixed_diagnostic');joint.CreateBody1Rel().SetTargets([Sdf.Path(bodypaths[1])]);joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float,d.xpos[1])));joint.CreateLocalRot0Attr(quat(d.xquat[1]))
 report={'converter':'explicit compiled MuJoCo model to USD','source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'bodies':audit,'filtered_geom_pairs':filtered,'allowed_geom_pairs':allowed,'ellipsoid_tessellation':[32,16],'contact_approximation':'isotropic Coulomb; wheel coefficient5 from explicit source pairs; other geom sliding coefficients with max combine; no anisotropic torsional/rolling coefficients or solref/solimp mapping','fixed_diagnostic':fixed}
 return m,report
