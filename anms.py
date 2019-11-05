'''
  File name: anms.py
  Author:
  Date created:
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed 
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''
import numpy as np
def anms(cimg, max_pts):

  H, W = cimg.shape
  # corner_threshold is designed to filter out
  # the corner we want to consider.
  # corner_threshold = 0.00001
  # up_range is only consider the filtered corner
  # in this range [1, up_range]
  up_range = 1.5

  # first, vectrize the ori cimg matrix
  vect_cimg = cimg.flatten()


  # set non-cornor element to 0
  filtered_vect_cimg = vect_cimg.copy()
  filtered_vect_cimg[filtered_vect_cimg < np.max(vect_cimg)*0.1] = 0
  if np.sum(filtered_vect_cimg) == 0:

    print("you choose the first threshold too high, change it to a smaller value")
  #
  coor_x_list = list()
  coor_y_list = list()
  r_min_list = list()
  # coor_r_matrix = np.array([np.inf,np.inf,np.inf]).reshape(-1,1)

  for indx, element in enumerate(filtered_vect_cimg):
    if element == 0:
      continue
    else:
      # record the current coor
      coor_i = indx//W
      coor_j = indx % W
      current_coor = (coor_i, coor_j)
      # add the x,y coor in
      coor_y_list.append(coor_i)
      coor_x_list.append(coor_j)

      # filter the compared points' coor
      logic = \
        np.logical_and(1 * element < filtered_vect_cimg, \
                       filtered_vect_cimg <= up_range*element)
      # find the index of these compare
      compare_indx = np.array(np.where(logic)).flatten()
      if len(compare_indx)>0:
        compare_coor_list = []
        for comp_indx in compare_indx:
          comp_coor_i = comp_indx // W
          comp_coor_j = comp_indx % W
          compare_coor_list.append((comp_coor_i, comp_coor_j))
        dist_list = np.linalg.norm(np.array(compare_coor_list) - np.array(current_coor), axis=1)
        min_dist = np.min(dist_list)

        r_min_list.append(min_dist)
      # cannot find the larger point in the given range
      else:
        r_min_list.append(np.inf)
  # find the sort order in the r_min_list, decreasing
  number_corner = len(r_min_list)
  order = sorted(range(number_corner), reverse=True, key=lambda k: r_min_list[k])
  # to avoid num exceed the dimension
  if number_corner <= max_pts:
    max_pts = number_corner
  x = np.array([coor_x_list[i] for i in order]).reshape(-1,1)[0:max_pts]
  y = np.array([coor_y_list[i] for i in order]).reshape(-1,1)[0:max_pts]
  rmax = np.min(np.array([r_min_list[i] for i in order]).reshape(-1,1)[0:max_pts])

  return x, y, rmax


if __name__ == '__main__':
    cimg = np.array([[1,1.01],[1.1,0]])
    max_pts = 100
    x,y,rmax = anms(cimg, max_pts)