import argparse
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt

def load_data(dir):
    imgs = []
    raws = []
    shutter = []
    for f in np.sort(os.listdir(dir)):
        if('jpg' in f or 'JPG' in f):
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

def sample_pixels(imgs, num_samples=100):

    x_idx = random.sample(range(imgs[0].shape[0]), num_samples)
    y_idx = random.sample(range(imgs[0].shape[1]), num_samples)
   
    res = np.zeros((num_samples, len(imgs), 3))
    for c in range(3):
        for p in range(len(imgs)):
            k = 0
            for x in x_idx:
                for y in y_idx:
                    res[k][p][c] = imgs[p][x, y, c]


    return res


def recoverResponseCurve(imgs, log_shutter, W, l, num_samples):
    response_curve = []
    Zs = sample_pixels(imgs, num_samples)
    # print(a)
    print(Zs.shape)
    
    n = num_samples
    p = len(imgs)

    for c in range(3):
        Z = np.array(Zs[:,:,c])
        print(np.array(Z).shape)
        A = np.zeros([n*p + 255, 256 + n])
        B = np.zeros((A.shape[0], 1))
        
        k = 0
        for i in range(n):
            for j in range(p):
                # print(i,j,Z[i][j])
                wij = W[int(Z[i][j])]
                A[k, int(Z[i][j])] = wij
                A[k, 256+i] = -wij
                B[k, 0] = wij * log_shutter[j]
                k += 1
   
        for i in range(1, 255):
            wi = W[i]
            A[k, i-1] = wi * l
            A[k, i] = -2 * wi * l
            A[k, i+1] = wi * l
            k += 1

        A[-1, 127] = 1

        A_inv = np.linalg.pinv(A)
        g = np.dot(A_inv, B)[:256].flatten()
        response_curve.append(g)
    return response_curve
        
def recoverRadianceMap(imgs, log_shutter, W, response_curve):
    radiance_map = np.zeros((imgs[0].shape[0], imgs[0].shape[1], 3))
    for c in range(3):
        for i in range(imgs[0].shape[0]):
            for j in range(imgs[0].shape[1]):
                sum_ = 0
                sum_w = 0
                for n in range(len(imgs)):
                    sum_ += W[imgs[n][i, j, c]] * (response_curve[c][imgs[n][i, j, c]] - log_shutter[n])
                    sum_w += W[imgs[n][i, j, c]]
                radiance_map[i, j, c] = sum_ / sum_w
    return np.exp(radiance_map)


def debevec(imgs, shutter, num_samples, out_dir):
    log_shutter = np.log(shutter)
    l = 10
    W = np.concatenate((np.arange(1, 129), np.arange(1, 129)[::-1]), axis=0)
    response_curve = recoverResponseCurve(imgs, log_shutter, W, l, num_samples)
    print(np.array(response_curve).shape)
    radiance_map = recoverRadianceMap(imgs, log_shutter, W, response_curve)
    

    cv2.imwrite(os.path.join(out_dir, 'radiance.hdr'), radiance_map)
    np.save(os.path.join(out_dir, 'radiance.npy'), radiance_map)
    # radiance_map = np.load(os.path.join(out_dir, 'radiance.npy'))

  

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
        debevec(imgs, shutter, 100, out_dir)
    elif args.hdr_method == 'fromraw':
        pass
    elif args.hdr_method == 'robertson':
        pass




