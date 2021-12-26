import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])

        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))
        image = config.both_transforms(image=image)["image"]
        target_imgs = config.target_transform(image=image)["image"]
        input_imgs = config.input_transform(image=image)["image"]
        return input_imgs, target_imgs


def test():
    dataset = MyImageFolder(root_dir=config.DATA_TRAIN)
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    for input_imgs, target_imgs in loader:
        print(input_imgs.shape)
        print(target_imgs.shape)


if __name__ == "__main__":
    test()