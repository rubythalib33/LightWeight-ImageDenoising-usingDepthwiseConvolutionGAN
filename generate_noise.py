import cv2
import os
from utils import add_noise

PATH = '../Datasets/DIV2K/DIV2K_valid_HR/'

for image in os.listdir(PATH):
    img = cv2.imread(os.path.join(PATH, image))
    img_noise = add_noise(img, 300000, 1000000)
    cv2.imwrite(os.path.join('noise_image', image), img_noise)