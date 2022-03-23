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

def save(hdr, ldr, args, response_curve, out_dir):
    np.save(os.path.join(out_dir, 'response_curve.npy'), response_curve)
    cv2.imwrite(os.path.join(out_dir, 'radiance.hdr'), hdr)
    np.save(os.path.join(out_dir, 'radiance.npy'), hdr)
    cv2.imwrite(os.path.join(out_dir, f'pho_global_{args.p_global}.png'), ldr)

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--hdr_method', type=str, default='debevec', choices=['debevec', 'fromraw', 'robertson'])
    return parser.parse_args()

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--hdr_method', type=str, default='debevec', choices=['debevec', 'fromraw', 'robertson'])
    parser.add_argument('--resize_ratio', type=float)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--smooth', type=int, default=20)
    parser.add_argument('--p_global', type=float, default=0.5)
    parser.add_argument('--p_local', type=float, default=0.5)
    # parser.add_argument('')
    return parser.parse_args()


if __name__ == '__main__':
    args = ParseArgs()
    out_dir = f"res_{args.data.split('/')[-1]}_{args.hdr_method}_{args.num_samples}_l{args.smooth}"
    if(args.resize_ratio):
        out_dir += f"_{args.resize_ratio}"
    print(out_dir)
    # print(a)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    imgs, raws, shutter = load_data(args.data)
    if(args.resize_ratio):
        imgs = np.array([cv2.resize(img, (int(img.shape[1]*args.resize_ratio), int(img.shape[0]*args.resize_ratio))) for img in imgs])
    print(imgs.shape ,raws.shape)
    print(shutter)
    
    if args.hdr_method == 'debevec':
        debevec = Debevec()
        hdr, response_curve = debevec.run(imgs, shutter, args.num_samples, args.smooth)
    elif args.hdr_method == 'fromraw':
        pass
    elif args.hdr_method == 'robertson':
        robert = Robertson()
        hdr, response_curve = robert.run(imgs, shutter)
    

    # tone mapping
    ldr = photographic_global_operator(hdr, args.p_global, 1e-8)


    plot_response_curve(response_curve, out_dir)
    save(hdr, ldr, args, response_curve, out_dir)
