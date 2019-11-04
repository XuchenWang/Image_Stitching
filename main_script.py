import numpy as np
from PIL import Image
from scipy import signal
from helper import rgb2gray
from corner_detector import corner_detector
from anms import anms

if __name__ == '__main__':
    file_name = "middle.jpg"
    # read in RGB matrix
    I = np.array(Image.open(file_name).convert('RGB'))
    # convert the RGB into gray image
    im_gray = rgb2gray(I)
    # find the corner matrix
    cimg = corner_detector(im_gray)

    # print(cimg)
    print(np.max(cimg[np.where(cimg > 0)]))

    # ANMS
    max_pts = 100
    x,y,rmax = anms(cimg, max_pts)


