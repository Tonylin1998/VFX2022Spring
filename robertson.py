import numpy as np
import cv2
import os

class Robertson():
    def __init__(self, imgs, shutter, out_dir):
        self.imgs = imgs
        self.shutter = shutter
        self.p = len(imgs)
        self.height = imgs[0].shape[0]
        self.width = imgs[0].shape[1]
        self.maxIter = 10
        self.out_dir = out_dir

        self.weight = np.array([128.5 -  abs(i - 127.5) for i in range(256)])
        self.G = [[(i - 128) * 0.9 / 128 + 1 for i in range(256)] for j in range(3)]
        self.cal_E()
        # for i in range(3):
        #     for j in range(100):
        #         self.G[i][j] /= 1.5
    

    def cal_E(self):
        newE = np.zeros((self.height, self.width, 3))
        for i in range(self.height):
            for j in range(self.width):
                for c in range(3):
                    nu, de = 0, 0
                    for k in range(self.p):
                        nu += self.weight[self.imgs[k][i, j, c]] * self.G[c][self.imgs[k][i, j, c]] * self.shutter[k]
                        de += self.weight[self.imgs[k][i, j, c]] * self.shutter[k] ** 2
                    newE[i, j, c] = nu / de
        self.E = newE
    
    def cal_G(self):
        newG = np.zeros((3, 256))
        count = np.zeros((3, 256))
        for k in range(self.p):
            for i in range(self.height):
                for j in range(self.width):
                    for c in range(3):
                        newG[c, self.imgs[k][i, j, c]] += self.shutter[k] * self.E[i, j, c]
                        count[c, self.imgs[k][i, j, c]] += 1
        
        self.G = newG / count
        for i in range(3):
            self.G[i] /= self.G[i][128]
    

    def run(self):
        for i in range(self.maxIter):
            self.cal_G()
            self.cal_E()
            self.save()
            print(f"Finish iteration {i}")

    def save(self):
        cv2.imwrite(os.path.join(self.out_dir, 'radiance.hdr'), self.E)
        np.save(os.path.join(self.out_dir, 'radiance.npy'), self.E)