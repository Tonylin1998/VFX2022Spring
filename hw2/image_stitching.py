import argparse
from fnmatch import translate
from nis import match
from unittest import result
import cv2
import numpy as np
import os, sys
import random
import matplotlib.pyplot as plt
from feature import *
from projection import *
from match import *

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--out_dir', type=str, default='./result')
    parser.add_argument('--resize_ratio', type=float)
    parser.add_argument('--align', action='store_true')
    parser.add_argument('--t', type=str)

    return parser.parse_args()

def load_data(dir):
    imgs = []
    focals = []
    for f in np.sort(os.listdir(dir)):
        if('jpg' in f or 'JPG' in f or 'png' in f or 'PNG' in f):
            if('pano' not in f):
                imgs.append(cv2.imread(os.path.join(dir, f)))
    with open(os.path.join(dir, 'pano.txt')) as f:
        cnt = 0 
        for line in f.readlines():
            cnt = (cnt + 1)%13
            if(cnt == 12):
                focals.append(float(line))
    
    tmp = []
    for i in range(len(focals)):
        if(focals[i] != 0):
            tmp.append(imgs[i])
    imgs = tmp
    focals = [f for f in focals if f != 0]

    return np.array(imgs), np.array(focals)

def get_result_size(h, w, translations):
    ox, oy = 0, 0
    min_x, min_y = 0, 0
    max_x, max_y = 0, 0
    
    for i in range(len(translations)):

        oy += translations[i][0]
        ox += translations[i][1]

        # min_x = min(ox, min_x)
        # max_x = max(ox, max_x)
        min_y = min(oy, min_y)
        max_y = max(oy, max_y)
    
    result_h = h + max_y - min_y
    result_w = w + np.sum(translations[:, 1])

    return result_h, result_w, -min_y

    



def blend(imgs, translations, align):
    n = len(imgs)
   
    if(translations[0][1] < 0):
        print('reverse')
        translations = -translations[::-1]
        imgs = imgs[::-1]
        if(align):
            translations = np.concatenate((translations[1:,:], [translations[0,:]]))

    if(align):
        base, extra = divmod(translations[-1][0], n-1)
        displacements = [base + (i < extra) for i in range(n-1)]
        for i in range(n-1):
            translations[i][0] += displacements[i]
    
    
    translations = translations[:-1, :]
    print(translations.shape)

    h, w, c = imgs[0].shape
    result_h, result_w, oy = get_result_size(h, w, translations)
    ox = 0

    imgs_weight = np.ones((n, h, w))
    for i in range(n-1):
        dy, dx = translations[i][0], translations[i][1]
        # print(dy, dx)
        if(dy >= 0):
            blend_w = w - dx
            blend_h = h - dy
            # blend_array = np.zeros((blend_h, blend_w))
            tmp = np.linspace(1, 0, blend_w)
            blend_array = np.tile(tmp, (blend_h, 1))

            imgs_weight[i][dy:, dx:] = blend_array
            imgs_weight[i+1][:blend_h, :blend_w] = 1-blend_array
        else:
            blend_w = w - dx
            blend_h = h - abs(dy)
            # blend_array = np.zeros((blend_h, blend_w))
            tmp = np.linspace(1, 0, blend_w)
            blend_array = np.tile(tmp, (blend_h, 1))

            imgs_weight[i][:blend_h, dx:] = blend_array
            imgs_weight[i+1][abs(dy):, :blend_w] = 1-blend_array

    
    
    for i in range(n):
        weight = np.zeros((h, w, 3))
        weight[:,:,0] = imgs_weight[i]
        weight[:,:,1] = imgs_weight[i]
        weight[:,:,2] = imgs_weight[i]

        result[oy:oy+h, ox:ox+w, :] += imgs[i] * weight
        
        if(i != n-1):
            oy += translations[i][0]
            ox += translations[i][1]
    # print(z)
    print(result.shape)
    return result.astype(np.uint8)

def image_stitching(imgs, focals, out_dir, args):
    print('=== start image stitching ===')
    n = len(imgs)

    print('--- cylindrical projection ---')
    imgs = cylindrical_projection(imgs, focals)

    if(args.t != None):
        translations = np.load(args.t)
    else:
        print('--- keypoints ---')
        keypoints = []
        for i in range(n):
            keypoints.append(harris_detector(imgs[i]))
        keypoints = np.array(keypoints)

        print('--- descriptors ---')
        descriptors = []
        for i in range(n):
            descriptors.append(keypoint_descriptor(imgs[i], keypoints[i]))
        descriptors = np.array(descriptors)

        print('--- matches ---')
        matches = []
        if(args.align):
            for i in range(n):
                matches.append(find_matches(descriptors[i], descriptors[(i+1)%n], 0.8))
        else:
            for i in range(n-1):
                matches.append(find_matches(descriptors[i], descriptors[i+1], 0.8))
        matches = np.array(matches)

        print('--- translations ---')
        translations = []
        if(args.align):
            for i in range(n):
                translations.append(ransac(matches[i], 1000, 3))
        else:
            for i in range(n-1):
                translations.append(ransac(matches[i], 1000, 3))
        translations = np.array(translations)
        np.save(os.path.join(out_dir, 'translations.npy'), translations)
    print(translations.shape)

    print('--- blending ---')
    result = blend(imgs, translations, args.align)



    return result

def crop(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)

    # print(np.size(img_thresh[0]))
    # print(np.size(np.where(img_thresh[0] != 0)))
    h, w, _ = image.shape
    low = 0
    upper = h-1
    for i in range(h):
        # print(np.size(np.where(img_thresh[i] != 0)))
        if np.size(np.where(img_thresh[i] != 0)) > 0.9*w:
            low = i
            break
    for i in range(h-1,-1,-1):
        if np.size(np.where(img_thresh[i] != 0)) > 0.9*w:
            upper = i+1
            break
    
    # print(low,upper)
    return image[low:upper]

if __name__ == '__main__':
    args = ParseArgs()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    out_dir = os.path.join(args.out_dir, f"{args.data.split('/')[-1]}")
    if(args.align):
        out_dir += '_align'


    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    imgs, focals = load_data(args.data)
    if(args.resize_ratio):
        imgs = np.array([cv2.resize(img, (int(img.shape[1]*args.resize_ratio), int(img.shape[0]*args.resize_ratio))) for img in imgs])
    

    print(imgs.shape)
    print(focals)

    pano = image_stitching(imgs, focals, out_dir, args)
    pano_crop = crop(pano)
    
    # cv2.namedWindow('pano', cv2.WINDOW_NORMAL)
    # cv2.imshow('pano', pano)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    cv2.imwrite(os.path.join(out_dir, 'pano.jpg'), pano)
    cv2.imwrite(os.path.join(out_dir, 'pano_crop.jpg'), pano_crop)

    