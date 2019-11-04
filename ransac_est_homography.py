'''
  File name: ransac_est_homography.py
  Author: Xuchen Wang
  Date created: Nov 3, 2019
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as 
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image. 
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''

import numpy as np
import random
from math import sqrt
from est_homography import est_homography

def ransac_est_homography(x1, y1, x2, y2, thresh): #x1, y1, x2, y2 should be numpy array
  k = 1000
  N = len(x1)
  H = np.zeros([3,3])
  inlier_ind = np.zeros([N,])
  for i in range(k):
    curr_inlier_ind = np.zeros([N,])
    sample = random.sample(list(range(N)), 4)
    x = x1[sample]
    y = y1[sample]
    X = x2[sample]
    Y = y2[sample]
    curr_H = est_homography(x, y, X, Y)

    for feature in range(N):  #need to be simplified
      pred_x = curr_H[0,0]*x1[feature] + curr_H[0,1]*y1[feature] + curr_H[0,2]
      pred_y = curr_H[1,0]*x1[feature] + curr_H[1,1]*y1[feature] + curr_H[1,2]
      pred_z = curr_H[2,0]*x1[feature] + curr_H[2,1]*y1[feature] + curr_H[2,2]
      pred_x = pred_x/pred_z
      pred_y = pred_y/pred_z
      dist = sqrt((pred_x-x2[feature])**2 + (pred_y-y2[feature])**2)
      if dist < thresh:
        curr_inlier_ind[feature] = 1
      else:
        curr_inlier_ind[feature] = 0

    if sum(curr_inlier_ind) > sum(inlier_ind):
      inlier_ind = curr_inlier_ind
      H = curr_H

  return H, inlier_ind
