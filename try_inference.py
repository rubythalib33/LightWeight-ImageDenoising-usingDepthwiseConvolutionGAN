import torch
import config
from model import Generator
import os
import numpy as np
from PIL import Image
from torchvision.utils import save_image

gen = Generator(in_channels=3).to(config.DEVICE)
checkpoint = torch.load(config.CHECKPOINT_GEN, map_location=config.DEVICE)
gen.load_state_dict(checkpoint["state_dict"])
input_imgs_folder = config.DATA_TEST
os.system("rm saved/*")
files = os.listdir(input_imgs_folder)
np.random.shuffle(files)
gen.eval()
for file in files[:10]:
    image = Image.open(input_imgs_folder + file)
    try:
        with torch.no_grad():
            upscaled_img = gen(
                config.test_transform(image=np.asarray(image))["image"]
                .unsqueeze(0)
                .to(config.DEVICE)
            )
        save_image(upscaled_img * 0.5 + 0.5, f"saved/{file.replace('.jpg', '_predicted.jpg')}")
        # image.save( f"saved/{file.replace('.jpg', '_groundtruth.jpg')}")
    except : print('Memory insufficient for that image')