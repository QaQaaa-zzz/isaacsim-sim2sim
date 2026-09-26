"""Compose labeled recorded-state videos; preserves their termination holds."""
from pathlib import Path
import imageio.v2 as iio,numpy as np
r=Path(__file__).resolve().parent/'results/sim2sim_media';names=['mjx_original','mjx_dr','nominal_original','nominal_dr'];readers=[iio.get_reader(r/n/'replay.mp4') for n in names]
with iio.get_writer(r/'comparison.mp4',fps=25,codec='libx264',quality=8,macro_block_size=16) as out:
 for k in range(75):
  imgs=[rd.get_data(k) for rd in readers];im=np.concatenate([np.concatenate(imgs[:2],axis=1),np.concatenate(imgs[2:],axis=1)],axis=0);out.append_data(im)
  if k==24:iio.imwrite(r/'comparison.png',im)
for rd in readers:rd.close()
