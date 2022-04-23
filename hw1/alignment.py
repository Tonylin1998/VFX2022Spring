import cv2
import numpy as np


class MTB():
    def image_shrink_2(self, img):
        return cv2.resize(img, (int(img.shape[0]*0.5), int(img.shape[1]*0.5)))

    def compute_bitmaps(self, img):
        median = np.median(img)

        tb = np.copy(img)
        eb = np.copy(img)

        tb[tb > median] = 255
        tb[tb < median] = 0
       
        eb[eb > median+5] = 255
        eb[eb < median-5] = 0
    
        return tb, eb

    def get_exp_shift(self, img1, img2, shift_bits):
        
        if shift_bits > 0:
            sml_im1 = self.image_shrink_2(img1)
            sml_im2 = self.image_shrink_2(img2)
            cur_shift = self.get_exp_shift(sml_im1, sml_im2, shift_bits-1)
            cur_shift[0] *= 2
            cur_shift[1] *= 2
        else:
            cur_shift = [0, 0]
            
        tb1, eb1 = self.compute_bitmaps(img1)
        tb2, eb2 = self.compute_bitmaps(img2)
        min_err = img1.shape[0] * img1.shape[1]
        shift_ret = [0, 0]

        
        for i in range(-1, 2):
            for j in range(-1, 2):
                xs = cur_shift[0] + i
                ys = cur_shift[1] + j
                shifted_tb2 = self.shift_image(tb2, xs, ys)
                shifted_eb2 = self.shift_image(eb2, xs, ys)
                diff_b = np.bitwise_xor(tb1, shifted_tb2)
                diff_b = np.bitwise_and(diff_b, eb1)
                diff_b = np.bitwise_and(diff_b, shifted_eb2)
                err = np.sum(diff_b)
                if err < min_err:
                    shift_ret[0] = xs
                    shift_ret[1] = ys
                    min_err = err
                    
        return shift_ret

    def shift_image(self, img, x, y):
        num_rows, num_cols = img.shape[:2]
        translation_matrix = np.float32([ [1, 0, x], [0, 1, y] ])
        return cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))

    def process(self, imgs, shift_bits):
        if len(imgs) <= 1:
            return imgs
        gray_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
       
        for i in range(1, len(imgs)):
            shift = self.get_exp_shift(gray_imgs[0], gray_imgs[i], shift_bits)
            imgs[i] = self.shift_image(imgs[i], shift[0], shift[1])
        return imgs
