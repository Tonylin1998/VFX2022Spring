import argparse
from urllib import response
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from robertson import Robertson


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


def sample_pixels(imgs, num_samples=100):
    idx = random.sample(range(imgs[0].shape[0] * imgs[0].shape[1]), num_samples)

    res = np.zeros((num_samples, len(imgs), 3))
    for k in range(num_samples):
        for p in range(len(imgs)):
            for c in range(3):
                res[k][p][c] = imgs[p][int(idx[k] / imgs[0].shape[1]), int(idx[k] % imgs[0].shape[1]), c]


    return res


def recoverResponseCurve(imgs, log_shutter, W, l, num_samples):
    response_curve = []
    Zs = sample_pixels(imgs, num_samples)
    # print(Zs.shape)
    # print(a)
    # print(Zs[0][0])
    
    n = num_samples
    p = len(imgs)

    for c in range(3):
        Z = np.array(Zs[:,:,c])
        # Z = np.array(Zs[c])
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
        A[k, 127] = 1
        k += 1
   
        for i in range(1, 255):
            wi = W[i]
            A[k, i-1] = wi * l
            A[k, i] = -2 * wi * l
            A[k, i+1] = wi * l
            k += 1

       
        # np.save('A.npy', A)
        # np.save('B.npy', B)

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
                # radiance_map[i, j, c] = (response_curve[c][imgs[len(imgs)//2][i, j, c]] - log_shutter[len(imgs)//2])
    return np.exp(radiance_map)

def plot_response_curve(response_curve, out_dir):
    colors = ['blue', 'green', 'red']
    for c in range(3):
        plt.plot(response_curve[c], np.arange(256), c=colors[c])
    plt.xlabel('Log Exposure')
    plt.ylabel('Pixel Value')
    plt.savefig(os.path.join(out_dir, 'response_curve.png'))


def debevec(imgs, shutter, num_samples, out_dir):
    log_shutter = np.log(shutter)
    l = 10
    W = np.array(list(range(1,129))+list(range(129,1,-1)))
    

    response_curve = recoverResponseCurve(imgs, log_shutter, W, l, num_samples)
    print(np.array(response_curve).shape)
    np.save(os.path.join(out_dir, 'response_curve.npy'), response_curve)
    # response_curve = np.load(os.path.join('res_jiufen_2_debevec/response_curve.npy'))

    plot_response_curve(response_curve, out_dir)
    radiance_map = recoverRadianceMap(imgs, log_shutter, W, response_curve)
    
    cv2.imwrite(os.path.join(out_dir, 'radiance.hdr'), radiance_map)
    np.save(os.path.join(out_dir, 'radiance.npy'), radiance_map)
    # radiance_map = np.load(os.path.join(out_dir, 'radiance.npy'))

    return radiance_map

def photographic_global_operator(hdr, a):
    Lw = np.exp(np.mean(np.log(1e-8 + hdr)))
    Lm = (a / Lw) * hdr
    L_white = np.max(Lm) 
    Ld = (Lm * (1 + (Lm/(L_white ** 2)))) / (1 + Lm)
    ldr = np.array(Ld * 255)
    
    return ldr


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
        hdr = debevec(imgs, shutter, 50, out_dir)
    elif args.hdr_method == 'fromraw':
        pass
    elif args.hdr_method == 'robertson':
        robert = Robertson(imgs, shutter, out_dir)
        robert.run()
        hdr = robert.E

    ldr = photographic_global_operator(hdr, 0.5)
    cv2.imwrite(os.path.join(out_dir, 'pho_global.png'), ldr)


