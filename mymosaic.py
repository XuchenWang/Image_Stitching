'''
  File name: mymosaic.py
  Author:
  Date created:
'''

'''
  File clarification:
    Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. If you want to implement
    imwarp (or similar function) by yourself, you should apply bilinear interpolation when you copy pixel values. 
    As a bonus, you can implement smooth image blending of the final mosaic.
    - Input img_input: M elements numpy array or list, each element is a input image.
    - Outpuy img_mosaic: H × W × 3 matrix representing the final mosaic image.
'''
import numpy as np
from helper import rgb2gray
from corner_detector import corner_detector
from anms import anms
from feat_desc import feat_desc
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography

def mymosaic(img_input):
  # Your Code Here
  img_input = list(img_input)
  N = len(img_input)
  img1 = img_input[0]
  for i in range(1,N):
    img2 = img_input[i]
    img1 = oneMosaic(img1,img2)

  img_mosaic = img1
  return img_mosaic


def oneMosaic(img1, img2): # input should be gray images
  im_gray1 = rgb2gray(img1)
  im_gray2 = rgb2gray(img2)
  cimg1 = corner_detector(im_gray1)
  cimg2 = corner_detector(im_gray2)

  max_pts1 = 50 #need to be adjust to different images
  x1,y1,rmax1 = anms(cimg1, max_pts1)
  max_pts2 = 100
  x2,y2,rmax2 = anms(cimg2, max_pts2)

    # feat_desc
  descs1 = feat_desc(im_gray1, x1.flatten(), y1.flatten())
  descs2 = feat_desc(im_gray2, x2.flatten(), y2.flatten())

    # feat_match
  final_match1 = []
  final_match2 = []
  match1 = feat_match(descs1,descs2)
  match2 = feat_match(descs2,descs1)
  for ind,value in enumerate(match1):
      if value!=-1 and match2[int(value)] == ind:
          final_match1.append(ind)
          final_match2.append(int(value))


    # ransac
  thresh = 0.5
  x1 = x1[final_match1]
  # print(type())
  y1 = y1[final_match1]
  x2 = x2[final_match2]
  y2 = y2[final_match2]
  H, inlier_ind = ransac_est_homography(x1, y1, x2, y2, thresh)


  #mosaicing
  Himg, Wimg, _ = img2.shape
  # indicate 4 augmented corner point in img2
  four_corner_1 = np.array([0,0,1])
  four_corner_2 = np.array([0,Wimg-1,1])
  four_corner_3 = np.array([Himg-1,0,1])
  four_corner_4 = np.array([Himg-1,Wimg-1,1])
  four_corner_array = np.vstack((four_corner_1,four_corner_2,four_corner_3,four_corner_4)).T
  assert four_corner_array.shape == (3,4)
  # get mapped 4 points in the img1
  mapped_points = H @ four_corner_array
  mapped_points_norm = mapped_points/mapped_points[2,]
  # mashgrid to get all the coor in a 4 sided area
  x_min = np.min(mapped_points_norm)
  x_max =
  y_min =
  y_max =





  resultImage = img1.copy()
  np.pad(resultImage, ((H, H), (W, W)), 'constant')




  return resultImage
