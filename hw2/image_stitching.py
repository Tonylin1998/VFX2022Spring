import argparse
from fnmatch import translate
from nis import match
from unittest import result
import cv2
import numpy as np
import os, sys
import random
import matplotlib.pyplot as plt
from features import HarrisCornerDetector, SIFTDescriptors, FeatureMatcher
from projection import CylindricalProjection
from image_align import PairwiseAlignment, crop

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
    # with open(os.path.join(dir, 'pano.txt')) as f:
    #     cnt = 0 
    #     for line in f.readlines():
    #         print(line)
    #         cnt = (cnt + 1)%13
    #         if(cnt == 12):
    #             focals.append(float(line))
    # focals = [696, 679, 670, 683, 675]
    with open(os.path.join(dir, 'pano.txt')) as f:
        lines = f.readlines()
    for i in range(1, len(lines)-1):
        if lines[i-1]=='\n' and lines[i+1]=='\n':
            focals += [float(lines[i])]
    
    tmp = []
    for i in range(len(focals)):
        if(focals[i] != 0):
            tmp.append(imgs[i])
    imgs = tmp
    focals = [f for f in focals if f != 0]

    return np.array(imgs), np.array(focals)

def image_stitching(imgs, focals, out_dir, args):
    print('=== start image stitching ===')
    n = len(imgs)

    print('--- cylindrical projection ---')
    imgs = CylindricalProjection().project(imgs, focals)

    if(args.t != None):
        translations = np.load(args.t)
    else:
        print('--- keypoints ---')
        keypoints = []
        for i in range(n):
            keypoints.append(HarrisCornerDetector().detect_key_points(imgs[i]))
        # keypoints = np.array(keypoints)

        print('--- descriptors ---')
        descriptors = []
        for i in range(n):
            descriptors.append(SIFTDescriptors().get_descriptors(imgs[i], keypoints[i]))
        # descriptors = np.array(descriptors)

        # print('--- matches ---')
        # matches = []
        # if(args.align):
        #     for i in range(n):
        #         matches.append(FeatureMatcher().find_matches(keypoints[i], keypoints[(i+1)%n], descriptors[i], descriptors[(i+1)%n]))
        # else:
        #     for i in range(n-1):
        #         matches.append(FeatureMatcher().find_matches(keypoints[i], keypoints[i+1], descriptors[i], descriptors[i+1]))
        # # matches = np.array(matches)

        print('--- get translations (ransac) ---')
        translations = []
        if(args.align):
            for i in range(n):
                translations.append(FeatureMatcher().get_translation(keypoints[i], keypoints[(i+1)%n], descriptors[i], descriptors[(i+1)%n], 10000))
        else:
            for i in range(n-1):
                translations.append(FeatureMatcher().get_translation(keypoints[i], keypoints[i+1], descriptors[i], descriptors[i+1], 10000))
        translations = np.array(translations)
        np.save(os.path.join(out_dir, 'translations.npy'), translations)

    print('--- blending ---')
    result = PairwiseAlignment().blend(imgs, translations, args.align)

    return result

if __name__ == '__main__':
    args = ParseArgs()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    out_dir = os.path.join(args.out_dir, f"{args.data.split('/')[-1]}")
    if(args.align):
        out_dir += '_align'
    if(args.resize_ratio):
        out_dir += f'_resize{args.resize_ratio}'


    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    imgs, focals = load_data(args.data)
    if(args.resize_ratio):
        imgs = np.array([cv2.resize(img, (int(img.shape[1]*args.resize_ratio), int(img.shape[0]*args.resize_ratio))) for img in imgs])
    

    print(imgs.shape)
    print(focals)
    # for img in imgs:
    #     cv2.namedWindow('pano', cv2.WINDOW_NORMAL)
    #     cv2.imshow('pano', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    pano = image_stitching(imgs, focals, out_dir, args)
    pano_crop = crop(pano)
    
    cv2.imwrite(os.path.join(out_dir, 'pano.jpg'), pano)
    cv2.imwrite(os.path.join(out_dir, 'pano_crop.jpg'), pano_crop)

    