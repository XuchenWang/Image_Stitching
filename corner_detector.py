'''
  File name: corner_detector.py
  Author:Lishuo Pan
  Date created: Nov 3, 2019
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''
from cv2 import cornerHarris
def corner_detector(img):
  # Your Code Here
  # print(type(img[0,0]))
  cimg = cornerHarris(img, 10, 3, 0.001)
  return cimg
