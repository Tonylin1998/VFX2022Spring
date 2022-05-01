import numpy as np
import cv2
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cdist



class HarrisCornerDetector():
    def detect_key_points(self, img, k=0.05, threshold=50000):
        kernel_size = 5

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        I = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
        Iy, Ix = np.gradient(I)

        Ix2 = Ix**2
        Iy2 = Iy**2
        Ixy = Ix*Iy

        Sx2 = cv2.GaussianBlur(Ix2, (kernel_size, kernel_size), 0)
        Sy2 = cv2.GaussianBlur(Iy2, (kernel_size, kernel_size), 0)
        Sxy = cv2.GaussianBlur(Ixy, (kernel_size, kernel_size), 0)

        R = (Sx2 * Sy2 - Sxy * Sxy) - k * (Sx2 + Sy2) ** 2

        threshold = 0.01 * np.max(R)
        R[R<threshold] = 0
        localMaxR = maximum_filter(R, size=3, mode='constant')
        R[R<localMaxR] = 0
        
    
        keypoints = np.array(np.where(R > 0)).T
        

        # keypoints = np.array(np.where(R > threshold)).T
        print(keypoints.shape)

        return keypoints

class SIFTDescriptors():
    def orientation_histogram(self, img, bins, sigma):
        kernel_size = 5
        h, w, _ = img.shape

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        I = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
        Iy, Ix = np.gradient(I)

        magnitude = np.sqrt(Ix**2 + Iy**2)
        theta = np.arctan2(Iy, Ix) * 180 / np.pi
        theta[theta<0] = theta[theta<0]+360
        
        binSize = 360/bins
        bucket = np.round(theta/binSize)
        histogram = np.zeros((bins,) + (h,w))  # (bins, h, w)
        for b in range(bins):
            histogram[b][bucket==b] = 1
            histogram[b] *= magnitude
            histogram[b] = cv2.GaussianBlur(histogram[b], (5,5), sigma)
        
        return histogram

    def get_descriptors(self, img, keypoints):
        histogram = self.orientation_histogram(img, bins=36, sigma=1.5)
        orientations_assignment = np.argmax(histogram, axis=0)*10 + 5
        
        descriptors = []
        h, w, _ = img.shape
        for kp in keypoints:
            i, j = kp[0], kp[1]

            rotation_matrix = cv2.getRotationMatrix2D((float(j), float(i)), orientations_assignment[i, j], 1)
            rotated = cv2.warpAffine(img, rotation_matrix, (w, h))

            histogram = self.orientation_histogram(rotated, bins=8, sigma=8)
            desc = []
            for di in range(i-8, i+4+1, 4):
                for dj in range(j-8, j+4+1, 4):
                    sub_feature = [0 for _ in range(8)]
                    i_low = max(0, di)
                    j_low = max(0, dj)
                    i_high = min(h, di+4) 
                    j_high = min(w, dj+4) 
                    for bin in range(8):
                        sub_feature[bin] = np.sum(histogram[bin][i_low:i_high, j_low:j_high])
                    desc += sub_feature

            desc /= np.linalg.norm(desc)
            descriptors.append(np.array(desc).astype('float32'))

        return np.array(descriptors)
        
        

class FeatureMatcher():
    def find_matches(self, kp1, kp2, des1, des2):
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        print(len(matches))
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        print(len(good_matches))

        good_matches = sorted(good_matches, key=lambda x: x.distance)
        # good_matches = delete_closer(good_matches, kp1, k)
        
        # good_matches = g ood_matches[:k]
        points1 = np.array([kp1[m.queryIdx] for m in good_matches])
        points2 = np.array([kp2[m.trainIdx] for m in good_matches])

        return points1, points2

    def get_translation(self, kp1, kp2, des1, des2, ransac_iter):
        points1, points2 = self.find_matches(kp1, kp2, des1, des2)
        
        max_cnt = -1
        for _ in range(ransac_iter):
            idx = np.random.randint(0, len(points1))
            translation = np.subtract(points1[idx], points2[idx])

            pred_points2 = points1 - translation
            cnt = 0
            for i in range(len(points2)):
                error = ((points2[i][0]-pred_points2[i][0])**2 + (points2[i][1]-pred_points2[i][1])**2)**(1/2)
                if(error < 3):
                    cnt += 1

            if(cnt > max_cnt):
                # print(cnt)
                max_cnt = cnt
                best_translation = translation
        return best_translation

