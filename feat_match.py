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

from PIL import Image
from helper import rgb2gray
from corner_detector import corner_detector
from feat_desc import feat_desc
import matplotlib.pyplot as plt

def feat_match(descs1, descs2):
  N1 = descs1.shape[1]
  print(N1)
  N2 = descs2.shape[1]
  print(N2)
  match = np.zeros([N1,])

  for i in range(N1):
  # for i in range(5):  #testing
  #   print('==========N1',i)
    match_dist = np.zeros([N2,])
    curr_descs1 = descs1[:,i]

    for j in range(N2):
    # for j in range(5):  #testing
      curr_descs2 = descs2[:,j]
      d = np.linalg.norm(curr_descs1-curr_descs2)
      # print('mean',d.mean())
      # print('std',d.std())
      match_dist[j] = d

    # print('match_dist: ', match_dist[:10])
    minValue = min(match_dist)
    minIndex = np.argmin(match_dist)
    # match_dist = np.delete(match_dist, minValue)
    match_dist = list(value for value in match_dist if value != minValue)
    second_minValue = min(match_dist)
    # print('two Value:', minValue,  second_minValue)
    if (minValue/(second_minValue+0.0000001)) < 0.6: # could also be 0.6
      match[i] = minIndex
    else:
      match[i] = -1

  # print('match: ', match)
  return match







#
# I1 = np.array(Image.open("test.jpg").convert('RGB'))
# I2 = np.array(Image.open("test1.jpg").convert('RGB'))
# im_gray1 = rgb2gray(I1)
# im_gray2 = rgb2gray(I2)
# cimg1 = corner_detector(im_gray1)
# cimg2 = corner_detector(im_gray2)
# cimg1[cimg1<0.5*cimg1.max()]=0
# cimg2[cimg2<0.5*cimg2.max()]=0
# # print(cimg)
# y1,x1 = np.nonzero(cimg1)
# y2,x2 = np.nonzero(cimg2)
# # I1[cimg1>0.01*cimg1.max()]=[0,0,255]
# # plt.imshow(I1)
# # plt.show()
# # I2[cimg2>0.01*cimg2.max()]=[0,0,255]
# # plt.imshow(I2)
# # plt.show()
# # print(im_gray.shape)
# # print(len(x))
# # print(len(y))
# descs1 = feat_desc(im_gray1, x1, y1)
# descs2 = feat_desc(im_gray2, x2, y2)
#
# match = feat_match(descs1, descs2)
# print(match[:10])



