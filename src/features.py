import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.stats import entropy

# -----------------------------
# Constants
# -----------------------------
LBP_POINTS = 8
LBP_RADIUS = 1

IMAGE_SIZE = 256


class FeatureExtractor:

    def __init__(self):
        pass

    def load_image(self, image_path):

        img = cv2.imread(image_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        return img, gray


    # -----------------------------
    # Sharpness
    # -----------------------------

    def laplacian_variance(self, gray):

        return cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()


    # -----------------------------
    # Edge Density
    # -----------------------------

    def edge_density(self, gray):

        edges = cv2.Canny(
            gray,
            100,
            200
        )

        return np.mean(edges > 0)


    # -----------------------------
    # Entropy
    # -----------------------------

    def image_entropy(self, gray):

        hist = cv2.calcHist(
            [gray],
            [0],
            None,
            [256],
            [0,256]
        ).flatten()

        hist /= hist.sum()

        return entropy(hist + 1e-8)


    # -----------------------------
    # HSV Histogram
    # -----------------------------

    def hsv_features(self, img):

        hsv = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2HSV
        )

        hist = cv2.calcHist(
            [hsv],
            [0,1],
            None,
            [8,8],
            [0,180,0,256]
        )

        hist = cv2.normalize(hist, hist).flatten()

        return hist


    # -----------------------------
    # Local Binary Pattern
    # -----------------------------

    def lbp_features(self, gray):

        lbp = local_binary_pattern(
            gray,
            P=LBP_POINTS,
            R=LBP_RADIUS,
            method="uniform"
        )

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0,11),
            range=(0,10)
        )

        hist = hist.astype(float)

        hist /= hist.sum()

        return hist


    # -----------------------------
    # FFT Features
    # -----------------------------

    def fft_features(self, gray):

        f = np.fft.fft2(gray)

        fshift = np.fft.fftshift(f)

        magnitude = np.log(
            np.abs(fshift)+1
        )

        return np.array([
            magnitude.mean(),
            magnitude.std(),
            magnitude.max()
        ])


    # -----------------------------
    # Specular Highlight
    # -----------------------------

    def glare_score(self, img):

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

        return np.mean(gray > 240)


    # -----------------------------
    # Main Function
    # -----------------------------

    def extract(self, image_path):

        img, gray = self.load_image(image_path)

        features = []

        features.append(
            self.laplacian_variance(gray)
        )

        features.append(
            self.edge_density(gray)
        )

        features.append(
            self.image_entropy(gray)
        )

        features.append(
            self.glare_score(img)
        )

        features.extend(
            self.fft_features(gray)
        )

        features.extend(
            self.lbp_features(gray)
        )

        features.extend(
            self.hsv_features(img)
        )

        return np.array(features, dtype=np.float32)