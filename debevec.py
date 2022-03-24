import numpy as np
import cv2
import os
import random

class Debevec():

    def sample_pixels(self, imgs, num_samples=100):
        idx = random.sample(range(imgs[0].shape[0] * imgs[0].shape[1]), num_samples)

        res = np.zeros((num_samples, len(imgs), 3))
        for k in range(num_samples):
            for p in range(len(imgs)):
                for c in range(3):
                    res[k][p][c] = imgs[p][int(idx[k] / imgs[0].shape[1]), int(idx[k] % imgs[0].shape[1]), c]


        return res

    def recoverRadianceMap(self, imgs, log_shutter, W, response_curve):
        radiance_map = np.zeros((imgs[0].shape[0], imgs[0].shape[1], 3))
        for c in range(3):
            # for i in range(imgs[0].shape[0]):
            #     for j in range(imgs[0].shape[1]):
            #         sum_ = 0
            #         sum_w = 0
            #         for n in range(len(imgs)):
            #             sum_ += W[imgs[n][i, j, c]] * (response_curve[c][imgs[n][i, j, c]] - log_shutter[n])
            #             sum_w += W[imgs[n][i, j, c]]
            #         radiance_map[i, j, c] = sum_ / sum_w
            #         # radiance_map[i, j, c] = (response_curve[c][imgs[len(imgs)//2][i, j, c]] - log_shutter[len(imgs)//2])

            sum_ = 0
            sum_w = 0
            for n in range(len(imgs)):
                img_flatten = imgs[n][:, :, c].flatten()
                sum_ += (response_curve[c][img_flatten] - log_shutter[n]) * W[img_flatten]
                sum_w += W[img_flatten]
            radiance_map[:, :, c] = (sum_ / sum_w).reshape(imgs[0].shape[0], imgs[0].shape[1])
        return np.exp(radiance_map)

    def recoverResponseCurve(self, imgs, log_shutter, W, l, num_samples):
        response_curve = []
        Zs = self.sample_pixels(imgs, num_samples)
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

    def run(self, imgs, shutter, num_samples, smooth):
        log_shutter = np.log(shutter)
        l = smooth
        W = np.array(list(range(1,129))+list(range(129,1,-1)))
        

        response_curve = self.recoverResponseCurve(imgs, log_shutter, W, l, num_samples)
        radiance_map = self.recoverRadianceMap(imgs, log_shutter, W, response_curve)
        print(np.array(response_curve).shape)
        # np.save(os.path.join(out_dir, 'response_curve.npy'), response_curve)
        # response_curve = np.load(os.path.join('res_jiufen_2_debevec/response_curve.npy'))

        # plot_response_curve(response_curve, out_dir)
        
        # cv2.imwrite(os.path.join(out_dir, 'radiance.hdr'), radiance_map)
        # np.save(os.path.join(out_dir, 'radiance.npy'), radiance_map)
        # radiance_map = np.load(os.path.join(out_dir, 'radiance.npy'))

        return radiance_map, response_curve