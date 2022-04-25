import numpy as np
import math
import cv2

def cylindrical_projection(images, focals):
    projection = np.zeros( (len(focals),) + images[0].shape, dtype=np.uint8)
    
    h, w, _ = images[0].shape
    for i in range(h):
        for j in range(w):
            x = j - int(w/2)
            y = h - 1 - i
            for f in range(len(focals)):
                theta = np.arctan(x / focals[f])
                height = y / np.sqrt(x ** 2 + focals[f] ** 2)
                px = int(focals[f] * theta) + int(w / 2)
                py = h - 1 - int(focals[f] * height)
                projection[f][py, px] = images[f][i, j]

    return projection
