import numpy as np
import cv2

class PairwiseAlignment():
    def get_result_size(self, h, w, translations):
        ox, oy = 0, 0
        min_x, min_y = 0, 0
        max_x, max_y = 0, 0
        
        for i in range(len(translations)):

            oy += translations[i][0]
            ox += translations[i][1]

            # min_x = min(ox, min_x)
            # max_x = max(ox, max_x)
            min_y = min(oy, min_y)
            max_y = max(oy, max_y)
        
        result_h = h + max_y - min_y
        result_w = w + np.sum(translations[:, 1])

        return result_h, result_w, -min_y

    def blend(self, imgs, translations, align):
        n = len(imgs)
        print(translations)
        if(translations[0][1] < 0):
            print('reverse')
            translations = -translations[::-1]
            imgs = imgs[::-1]
            if(align):
                translations = np.concatenate((translations[1:,:], [translations[0,:]]))
        # print(translations)
        if(align):
            base, extra = divmod(translations[-1][0], n-1)
            displacements = [base + (i < extra) for i in range(n-1)]
            # print(displacements)
            for i in range(n-1):
                translations[i][0] += displacements[i]

            translations = translations[:-1, :]
        # print(translations)
        
        
        
        # print(translations.shape)

        h, w, c = imgs[0].shape
        result_h, result_w, oy = self.get_result_size(h, w, translations)
        ox = 0

        imgs_weight = np.ones((n, h, w))
        for i in range(n-1):
            dy, dx = translations[i][0], translations[i][1]
            # print(dy, dx)
            if(dy >= 0):
                blend_w = w - dx
                blend_h = h - dy
                # blend_array = np.zeros((blend_h, blend_w))
                tmp = np.linspace(1, 0, blend_w)
                blend_array = np.tile(tmp, (blend_h, 1))

                imgs_weight[i][dy:, dx:] = blend_array
                imgs_weight[i+1][:blend_h, :blend_w] = 1-blend_array
            else:
                blend_w = w - dx
                blend_h = h - abs(dy)
                # blend_array = np.zeros((blend_h, blend_w))
                tmp = np.linspace(1, 0, blend_w)
                blend_array = np.tile(tmp, (blend_h, 1))

                imgs_weight[i][:blend_h, dx:] = blend_array
                imgs_weight[i+1][abs(dy):, :blend_w] = 1-blend_array
        
        result = np.zeros((result_h, result_w, c))
        
        for i in range(n):
            weight = np.zeros((h, w, 3))
            weight[:,:,0] = imgs_weight[i]
            weight[:,:,1] = imgs_weight[i]
            weight[:,:,2] = imgs_weight[i]
            result[oy:oy+h, ox:ox+w, :] += imgs[i] * weight

            
            if(i != n-1):
                oy += translations[i][0]
                ox += translations[i][1]
        print(np.max(result))
        return result.astype(np.uint8)


def crop(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    top, down = 0, img.shape[0]-1
    for i in range(img.shape[0]):
        if np.count_nonzero(img_gray[i]) > 0.99 * img.shape[1]:
            top = i
            break
    for i in range(img.shape[0]-1, -1, -1):
        if np.count_nonzero(img_gray[i]) > 0.99 * img.shape[1]:
            down = i
            break
    
    return img[top:down+1]