import numpy as np
import cv2
import os
# from hdr import photographic_global_operator

class Robertson():
    def __init__(self):
        self.maxIter = 10
        self.weight = np.array([np.exp(128.5 -  abs(i - 127.5)) for i in range(256)])
        # self.weight = np.array([1 for i in range(256)])
        # self.weight = np.array([max(256 - i, 128) for i in range(256)])
    

    def cal_E(self, imgs, shutter):
        p = len(imgs)
        height, width, _ = imgs[0].shape
        newE = np.zeros((height, width, 3))
        for i in range(height):
            for j in range(width):
                for c in range(3):
                    nu, de = 0, 0
                    for k in range(p):
                        nu += self.weight[imgs[k][i, j, c]] * self.G[c][imgs[k][i, j, c]] * shutter[k]
                        de += self.weight[imgs[k][i, j, c]] * shutter[k] ** 2
                    newE[i, j, c] = nu / de
        self.E = newE
    
    def cal_G(self, imgs, shutter):
        newG = np.zeros((3, 256))
        count = np.zeros((3, 256))
        p = len(imgs)
        height, width, _ = imgs[0].shape
        for k in range(p):
            for i in range(height):
                for j in range(width):
                    for c in range(3):
                        newG[c, imgs[k][i, j, c]] += shutter[k] * self.E[i, j, c]
                        count[c, imgs[k][i, j, c]] += 1
        
        self.G = newG / count
        for i in range(3):
            self.G[i] /= self.G[i][128]
    
    def cal_sum(self):
        return np.sum(self.E.flatten())

    def initEG(self, imgs, shutter):
        self.G = [[(i - 128) * 0.999 / 128 + 1 for i in range(256)] for j in range(3)]
        self.cal_E(imgs, shutter)


    def run(self, imgs, shutter):
        self.initEG(imgs, shutter)
        for i in range(self.maxIter):
            self.cal_G(imgs, shutter)
            lastSum = self.cal_sum()
            self.cal_E(imgs, shutter)
            curSum = self.cal_sum()
            # self.save()
            print(f"Finish iteration {i}, lastSum: {lastSum}, curSum: {curSum}")
            if abs(lastSum - curSum) / lastSum < 0.01:
                break
        return self.E, self.G