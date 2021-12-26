import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import SaltPapper

#First Train using all loss func until Epoch 176

DATA_TRAIN = '../Datasets/DIV2K/train_DIV2K/'
DATA_TEST = 'noise_image/'
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
START_EPOCHS = 1
NUM_EPOCHS = 200
BATCH_SIZE = 4
NUM_WORKERS = 4
IMAGE_SHAPE = 128
IMG_CHANNELS = 3

target_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

input_transform = A.Compose(
    [
        SaltPapper(low_n_frame=0.01, high_n_frame=0.1),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

both_transforms = A.Compose(
    [
        A.RandomCrop(width=IMAGE_SHAPE, height=IMAGE_SHAPE),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)