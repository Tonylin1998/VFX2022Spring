import numpy as np
import cv2

def photographic_global_operator(hdr, a, delta):
    Lw = np.exp(np.mean(np.log(delta + hdr)))
    Lm = (a / Lw) * hdr
    L_white = np.max(Lm) 
    Ld = (Lm * (1 + (Lm/(L_white ** 2)))) / (1 + Lm)
    ldr = np.array(Ld * 255)
    
    return ldr


def get_gaussian_filter(max_filter_width):
    filters = []
    for n in range(max_filter_width):
        f = np.array([[np.exp(-((i - n) ** 2 + (j - n) ** 2)) / np.pi for j in range(2 * n + 1)] for i in range(2 * n + 1)])
        filters.append(f / np.sum(f.flatten()))
    return filters

def get_padding_img(Lm, max_filter_width):
    h, w, _ = Lm.shape
    canvas = np.zeros((h + 2 * max_filter_width, w + 2 * max_filter_width, 3))
    canvas[max_filter_width: h + max_filter_width, max_filter_width: w + max_filter_width] = Lm
    # print(canvas[101][101])
    return canvas

def get_blur(padding_img, i, j, k, filter, s, max_filter_width):
    i += max_filter_width
    j += max_filter_width
    img = padding_img[i - s: i + s + 1, j - s: j + s + 1, k]
    return np.sum(np.multiply(filter, img))

def photographic_local_operator(hdr, a, delta):
    Lw = np.exp(np.mean(np.log(delta + hdr)))
    Lm = (a / Lw) * hdr
    ldr = np.zeros(hdr.shape)
    max_filter_width = 5
    filters = get_gaussian_filter(max_filter_width)
    padding_img = get_padding_img(Lm, max_filter_width)
    for i in range(hdr.shape[0]):
        for j in range(hdr.shape[1]):
            for k in range(3):
                for s in range(1, max_filter_width - 1):
                    cur = get_blur(padding_img, i, j, k, filters[s], s, max_filter_width)
                    nex = get_blur(padding_img, i, j, k, filters[s + 1], s + 1, max_filter_width)
                    if abs((cur - nex) / (a / s ** 2 + cur)) > 2e-16 or s == max_filter_width - 2:
                        # print(f"s={s}")
                        ldr[i][j][k] = Lm[i][j][k] / (1 + get_blur(padding_img, i, j, k, filters[s - 1], s - 1, max_filter_width))
    
    return ldr * 255