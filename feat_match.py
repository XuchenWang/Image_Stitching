'''
  File name: feat_match.py
  Author: Xuchen Wang
  Date created: Nov 3, 2019
'''

'''
  File clarification:
    Matching feature descriptors between two images. You can use k-d tree to find the k nearest neighbour. 
    Remember to filter the correspondences using the ratio of the best and second-best match SSD. You can set the threshold to 0.6.
    - Input descs1: 64 × N1 matrix representing the corner descriptors of first image.
    - Input descs2: 64 × N2 matrix representing the corner descriptors of second image.
    - Outpuy match: N1 × 1 vector where match i points to the index of the descriptor in descs2 that matches with the
                    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''

import numpy as np

def feat_match(descs1, descs2):
  N1 = descs1.shape[1]
  N2 = descs2.shape[1]
  match = np.zeros([N1,])

  for i in range(N1):
    match_dist = np.zeros([N2,])
    curr_descs1 = descs1[:,i]

    for j in range(N2):
      curr_descs2 = descs2[:,j]
      d = np.linalg.norm(curr_descs1-curr_descs2)
      match_dist[j] = d

    minValue = min(match_dist)
    minIndex = np.argmin(match_dist)
    match_dist = np.delete(match_dist, minValue)
    second_minValue = min(match_dist)
    if (minValue/second_minValue) < 0.7: # could also be 0.6
      match[i] = minIndex
    else:
      match[i] = -1

  return match












