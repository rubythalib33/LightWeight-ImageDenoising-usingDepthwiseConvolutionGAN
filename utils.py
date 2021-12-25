import torch
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import numpy as np
import random

from albumentations.core.transforms_interface import DualTransform


def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_examples(low_res_folder, gen):
    os.system("rm saved/*")
    files = os.listdir(low_res_folder)
    np.random.shuffle(files)
    gen.eval()
    for file in files[:10]:
        image = Image.open(low_res_folder + file)
        try:
            with torch.no_grad():
                upscaled_img = gen(
                    config.test_transform(image=np.asarray(image))["image"]
                    .unsqueeze(0)
                    .to(config.DEVICE)
                )
            save_image(upscaled_img * 0.5 + 0.5, f"saved/{file}")
        except : print('Memory insufficient for that image')
    gen.train()

def add_noise(img, low_percentage=200, high_percentage=500):
    # Getting the dimensions of the image
    row , col = img.shape[:2]
     
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(low_percentage, high_percentage)
    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to white
        img[y_coord][x_coord] = [255, 255, 255]
         
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(low_percentage , high_percentage)
    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to black
        img[y_coord][x_coord] = [0,0,0]
         
    return img

class SaltPapper(DualTransform):
    def __init__(self, low_n_frame, high_n_frame):
        super().__init__(True, 1.0)
        self.low_n_frame = low_n_frame
        self.high_n_frame = high_n_frame
    def apply(self, img):
        return add_noise(img, low_percentage=self.low_n_frame, high_percentage=self.high_n_frame)