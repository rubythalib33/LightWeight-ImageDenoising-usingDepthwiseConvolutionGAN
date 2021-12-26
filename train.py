import torch
import config
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from tqdm import tqdm
from dataset import MyImageFolder

torch.backends.cudnn.benchmark = True


def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):
    loop = tqdm(loader, leave=True)

    for idx, (input_imgs, target_imgs) in enumerate(loop):
        target_imgs = target_imgs.to(config.DEVICE)
        input_imgs = input_imgs.to(config.DEVICE)
        
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        fake = gen(input_imgs)
        disc_real = disc(target_imgs)
        disc_fake = disc(fake.detach())
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        
        loss_disc = disc_loss_fake + disc_loss_real
        # print(f'discrimantor loss:{loss_disc}')

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        disc_fake = disc(fake)
        l2_loss = mse(fake, target_imgs)
        adversarial_loss = bce(disc_fake, torch.ones_like(disc_fake))
        loss_for_vgg = vgg_loss(fake, target_imgs)
        # gen_loss = loss_for_vgg + adversarial_loss + l2_loss
        gen_loss = 4e-1 * loss_for_vgg + 2e-1 * adversarial_loss + 0.4*l2_loss
        # print(f'Generative loss:{gen_loss}')

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        if idx % 199 == 0 and idx != 0:
            plot_examples(config.DATA_TEST, gen)
            print(f'discrimantor loss:{loss_disc}')
            print(f'Generative loss:{gen_loss}')

def main():
    dataset = MyImageFolder(root_dir=config.DATA_TRAIN)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
           config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    for epoch in range(config.START_EPOCHS-1,config.NUM_EPOCHS):
        print(f'======================EPOCH: {epoch+1}=====================')
        train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()