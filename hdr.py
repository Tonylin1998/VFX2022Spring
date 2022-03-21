import argparse
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from robertson import Robertson
from debevec import Debevec
from toneMapping import *

def load_data(dir):
    imgs = []
    raws = []
    shutter = []
    for f in np.sort(os.listdir(dir)):
        if('jpg' in f or 'JPG' in f or 'png' in f):
            imgs.append(cv2.imread(os.path.join(dir, f)))
        elif('ARW' in f):
            pass
    with open(os.path.join(dir, 'shutter_times.txt')) as f:
        for line in f.readlines():
            if('/' in line):
                a = float(line.split('/')[0])
                b = float(line.split('/')[1])
                shutter.append(float(a/b))
            else:
                shutter.append(float(line))
    
    return np.array(imgs), np.array(raws), shutter

def plot_response_curve(response_curve, out_dir):
    colors = ['blue', 'green', 'red']
    for c in range(3):
        plt.plot(response_curve[c], np.arange(256), c=colors[c])
    plt.xlabel('Log Exposure')
    plt.ylabel('Pixel Value')
    plt.savefig(os.path.join(out_dir, 'response_curve.png'))

def save(hdr, ldr, response_curve, out_dir):
    np.save(os.path.join(out_dir, 'response_curve.npy'), response_curve)
    cv2.imwrite(os.path.join(out_dir, 'radiance.hdr'), hdr)
    np.save(os.path.join(out_dir, 'radiance.npy'), hdr)
    cv2.imwrite(os.path.join(out_dir, 'pho_global.png'), ldr)


def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--hdr_method', type=str, default='debevec', choices=['debevec', 'fromraw', 'robertson'])
    return parser.parse_args()

if __name__ == '__main__':
    args = ParseArgs()
    out_dir = f"res_{args.data.split('/')[-1]}_{args.hdr_method}"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    imgs, raws, shutter = load_data(args.data)
    imgs = np.array([cv2.resize(img, (int(img.shape[1]*0.2), int(img.shape[0]*0.2))) for img in imgs])
    print(imgs.shape ,raws.shape)
    print(shutter)
    
    if args.hdr_method == 'debevec':
        debevec = Debevec()
        hdr, response_curve = debevec.run(imgs, shutter, 50, out_dir)
    elif args.hdr_method == 'fromraw':
        pass
    elif args.hdr_method == 'robertson':
        robert = Robertson()
        hdr, response_curve = robert.run(imgs, shutter)
    


    ldr = photographic_global_operator(hdr, 0.5, 1e-8)
    plot_response_curve(response_curve, out_dir)
    save(hdr, ldr, response_curve, out_dir)
