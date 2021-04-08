import os

import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt

src = cv2.imread('examples' + os.sep + 'rotated' + os.sep + 'p3.png')

# apply guassian blur on src image
dst = cv2.GaussianBlur(src, (5, 5), cv2.BORDER_DEFAULT)

# display input and output image
cv2.imshow("Gaussian Smoothing", numpy.hstack((src, dst)))
cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image
