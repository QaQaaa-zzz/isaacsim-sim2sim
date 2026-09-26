"""Convex sectors whose union approximates the unchanged source ellipsoid."""
import numpy as np
from scipy.spatial import ConvexHull

def ellipsoid_sectors(size,sectors=16,polar=32,per_sector=4):
 result=[]
 for sector in range(sectors):
  phi=np.linspace(2*np.pi*sector/sectors,2*np.pi*(sector+1)/sectors,per_sector+1);theta=np.linspace(0,np.pi,polar+1);vertices=np.array([[np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)] for t in theta for p in phi])*np.asarray(size)
  vertices=np.unique(vertices.round(12),axis=0);h=ConvexHull(vertices);faces=h.simplices.copy()
  for i,face in enumerate(faces):
   a,b,c=vertices[face]
   if np.dot(np.cross(b-a,c-a),h.equations[i,:3])<0:faces[i]=face[[0,2,1]]
  result.append((vertices,faces))
 return result
