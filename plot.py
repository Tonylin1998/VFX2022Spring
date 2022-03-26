import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def load_data(dir):
    imgs = []
    raws = []
    shutter = []
    for f in np.sort(os.listdir(dir)):
        if('jpg' in f or 'JPG' in f or 'png' in f):
            imgs.append(cv2.imread(os.path.join(dir, f)))
        elif('ARW' in f):
            pass
    with open(os.path.join(dir, 'shutter.txt')) as f:
        for line in f.readlines():
            if('/' in line):
                a = float(line.split('/')[0])
                b = float(line.split('/')[1])
                shutter.append(float(a/b))
            else:
                shutter.append(float(line))
    
    return np.array(imgs), np.array(raws), shutter

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args = ParseArgs()
    
    imgs, raws, shutter = load_data(args.data)

    h, w = imgs[0].shape[0], imgs[0].shape[1]
    w /= h
    h = 1
    
    print(h,w)
    fig, ax = plt.subplots(3, 3, figsize=(3*w*5, 3*h*5))
    print(len(imgs))
    for i in range(len(imgs)): 
        x = i // 3
        y = i % 3

        # ax[x, y].imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
        ax[x, y].imshow(imgs[i], interpolation='nearest')
        ax[x, y].axis('off')
        ax[x, y].set_title(f'shutter time: {shutter[i]}', fontsize=20)
        
    fig.savefig(os.path.join('./report_img', f"{args.data.split('/')[-1]}_input.png"),  bbox_inches='tight')