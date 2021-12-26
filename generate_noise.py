import cv2
import os
import random
from tqdm import tqdm

PATH = '../Datasets/DIV2K/DIV2K_valid_LR_bicubic/X4/'
def add_noise(img, low_percentage=0.01, high_percentage=0.1):
    # Getting the dimensions of the image
    row , col = img.shape[:2]
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(int(row*col*low_percentage), int(row*col*high_percentage))
    for i in tqdm(range(number_of_pixels)):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to white
        img[y_coord][x_coord][:] = 255
         
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(int(row*col*low_percentage), int(row*col*high_percentage))
    for i in tqdm(range(number_of_pixels)):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to black
        img[y_coord][x_coord][:] = 0

    return img

for image in tqdm(os.listdir(PATH)):
    img = cv2.imread(os.path.join(PATH, image))
    img_noise = add_noise(img, 0.01, 0.1)
    cv2.imwrite(os.path.join('noise_image', image), img_noise)