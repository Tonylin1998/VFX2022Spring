import numpy as np
import cv2

def photographic_global_operator(hdr, a, delta):
    Lw = np.exp(np.mean(np.log(delta + hdr)))
    Lm = (a / Lw) * hdr
    L_white = np.max(Lm) 
    Ld = (Lm * (1 + (Lm/(L_white ** 2)))) / (1 + Lm)
    ldr = np.array(Ld * 255)
    
    return ldr


def get_gaussian_filter(n):
    f = np.array([[np.exp(-((i - n) ** 2 + (j - n) ** 2)) / np.pi for j in range(2 * n + 1)] for i in range(2 * n + 1)])
    return f / np.sum(f.flatten())

def get_padding_img(hdr, i, j, n):
    pad = 100
    h, w, _ = hdr.shape
    canvas = np.zeros((h + 2 * pad, w + 2 * pad, 3))
    canvas[pad: h + pad, pad: w + pad] = hdr
    # print(canvas[101][101])
    return canvas[pad + i - n: pad + i + n + 1, pad + j - n: pad + j + n + 1]

def get_blur(img, i, j, k, s):
    filter = get_gaussian_filter(s)
    pad = get_padding_img(img, i, j, s)
    return np.sum(np.multiply(filter, pad[:,:,k]))

def photographic_local_operator(hdr, a, delta):
    Lw = np.exp(np.mean(np.log(delta + hdr)))
    Lm = (a / Lw) * hdr
    ldr = np.zeros(hdr.shape)
    for i in range(hdr.shape[0]):
        for j in range(hdr.shape[1]):
            for k in range(3):
                for s in range(1, 20):
                    cur = get_blur(Lm, i, j, k, s)
                    nex = get_blur(Lm, i, j, k, s + 1)
                    if abs((cur - nex) / (a / s ** 2 + cur)) > 2e-16:
                        ldr[i][j][k] = Lm[i][j][k] / (1 + get_blur(Lm, i, j, k, s - 1))
    
    return ldr