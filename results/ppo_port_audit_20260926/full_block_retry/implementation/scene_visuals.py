"""Visual ground and authoritative ellipsoid wheel envelope; no new colliders."""
import numpy as np
from pxr import UsdGeom,Gf
from usd_model import mesh,transform

def enhance(stage,m):
 for g in range(m.ngeom):
  b=int(m.geom_bodyid[g]);typ=m.geom_type[g]
  if b and m.body(b).name in ['frontwheel','rearwheel'] and not(m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_rgba[g,3]>0:
   name=m.geom(g).name or f'geom_{g}';prim=stage.GetPrimAtPath(f'/World/Bike/{m.body(b).name}/{name}_g{g}');UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
 for body in ['frontwheel','rearwheel']:
  root=f'/World/Bike/{body}';g=m.geom(body+'_collision');size=g.size
  # Show the authoritative ellipsoid collision envelope. No invented tire width.
  vertices=[];faces=[];n=96;k=48
  for j in range(k+1):
   t=np.pi*j/k
   for i in range(n):
    p=2*np.pi*i/n;vertices.append(size*np.array([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)]))
  for j in range(k):
   for i in range(n):
    a=j*n+i;b=j*n+(i+1)%n;c=a+n;d=b+n
    if j:faces.append([a,b,c])
    if j<k-1:faces.append([b,d,c])
  tire=mesh(stage,root+'/Tire',np.asarray(vertices),np.asarray(faces));tire.CreateDisplayColorAttr([Gf.Vec3f(.025,.028,.033)])
  # Small colored hub caps are visual only and stay within the source envelope.
  for side in [-1,1]:
   hub=UsdGeom.Sphere.Define(stage,root+('/Hub_a' if side<0 else '/Hub_b'));hub.CreateRadiusAttr(1);transform(hub.GetPrim(),[0,0,side*size[2]*.90],[1,0,0,0],[size[0]*.16,size[1]*.16,size[2]*.035]);hub.CreateDisplayColorAttr([Gf.Vec3f(.65,.68,.7)])
 # Flat checker surface slightly above existing physics plane; no collision.
 verts=[];faces=[];colors=[]
 for x in range(-4,24):
  for y in range(-6,7):
   i=len(verts);xx=x*.5;yy=y*.5;verts.extend([[xx,yy,.0002],[xx+.5,yy,.0002],[xx+.5,yy+.5,.0002],[xx,yy+.5,.0002]]);faces.extend([[i,i+1,i+2],[i,i+2,i+3]]);c=(.24,.28,.32) if (x+y)%2 else (.36,.4,.44);colors.extend([Gf.Vec3f(*c)]*2)
 ground=mesh(stage,'/World/CheckerGround',np.array(verts),np.array(faces));ground.CreateDisplayColorAttr(colors);ground.GetDisplayColorPrimvar().SetInterpolation('uniform')
