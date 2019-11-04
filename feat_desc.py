'''
  File name: feat_desc.py
  Author: Xuchen Wang
  Date created: Nov 3, 2019
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature, 
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40 
    window to have a nice big blurred descriptor. 
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''

import numpy as np


def feat_desc(img, x, y):
  N = len(x)
  descs = np.zeros([64,N])
  img = np.pad(img, ((20, 20), (20, 20)), 'constant')
  x = x+20
  y = y+20
  Ig = img.astype(np.float64())
  [gradx, grady] = np.gradient(Ig)
  e = np.abs(gradx) + np.abs(grady)

  for i in range(N):
    x_cor = x[i]
    y_cor = y[i]
    desc_col = oneDesc(x_cor, y_cor, e)   #get one column for descs
    descs[:,i] = desc_col

  return descs

def oneDesc(x_cor, y_cor, e):
  desc_col = []
  x = x_cor-19
  y = y_cor-19
  for i in range(8):
    y = y + i*5
    for j in range(8):
      x = x + j*5
      max_mag = np.amax(e[y:y+5, x:x+5])
      desc_col.append(max_mag)

  desc_col = np.asarray(desc_col)
  desc_col = (desc_col - desc_col.mean()) / desc_col.std()
  return desc_col
