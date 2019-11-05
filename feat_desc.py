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

from PIL import Image
from helper import rgb2gray
from corner_detector import corner_detector

def feat_desc(img, x, y):
  N = len(x)
  # print(N)
  descs = np.zeros([64,N])
  img = np.pad(img, ((20, 20), (20, 20)), 'constant')
  # print(img.shape)
  x = x+20
  y = y+20
  Ig = img.astype(np.float64())
  [gradx, grady] = np.gradient(Ig)
  e = np.abs(gradx) + np.abs(grady)
  # print(e.shape)
  # print(e[0,:])

  for i in range(N):
    x_cor = x[i]
    y_cor = y[i]
    # if i < 2: #testing
    #   print("=========",i)
    desc_col = oneDesc(x_cor, y_cor, e)   #get one column for descs
    descs[:,i] = desc_col

  return descs

def oneDesc(x_cor, y_cor, e):
  # print('e-shape',e.shape)
  desc_col = []
  x_start = x_cor-19
  # print("x_start:", x_start)
  y_start = y_cor-19
  # print("y_start:", x_start)
  # print('.....')
  for i in range(8):
    y = y_start + i*5
    # print("y:", y)
    for j in range(8):
      x = x_start + j*5
      # print("x:", x)
      # print(e[y:y+5, x:x+5])
      max_mag = np.amax(e[y:y+5, x:x+5])
      desc_col.append(max_mag)

  desc_col = np.asarray(desc_col)
  desc_col = (desc_col - desc_col.mean()) / desc_col.std()
  return desc_col

#
# I = np.array(Image.open("test.jpg").convert('RGB'))
# im_gray = rgb2gray(I)
# cimg = corner_detector(im_gray)
# cimg[cimg<0.01*cimg.max()]=0
# # print(cimg)
# y,x = np.nonzero(cimg)
# # print(im_gray.shape)
# # print(len(x))
# # print(len(y))
# descs = feat_desc(im_gray, x, y)
# # print(descs.shape)
# # print(descs[:,-1])
