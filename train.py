import torch
from dataset import HorseZebraDataset
from utils import save_checkpoint, load_checkpoint
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_framework import Discriminator
from generator_framework import Generator
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast  # ✅ use torch.amp
import matplotlib as plt
import pandas as pd
import os


def train(disc_H, disc_Z, gen_Z, gen_H, loader, opt_gen, opt_disc, L1, mse, d_scaler, g_scaler, epoch,G_losses, D_losses):
    loop = tqdm(loader, leave=True)
    for idx, (zebra, horse) in enumerate(loop):
        print(f"Batch {idx} loaded. Zebra: {zebra.shape}, Horse: {horse.shape}")
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # ---------------------
        # Train Discriminators
        # ---------------------
        with autocast(device_type="cuda"):  # ✅ works on CPU + GPU
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            D_H_loss = mse(D_H_real, torch.ones_like(D_H_real)) + \
                       mse(D_H_fake, torch.zeros_like(D_H_fake))

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_loss = mse(D_Z_real, torch.ones_like(D_Z_real)) + \
                       mse(D_Z_fake, torch.zeros_like(D_Z_fake))

            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # ---------------------
        # Train Generators
        # ---------------------
        with autocast(device_type="cuda"):
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle losses
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_loss = L1(zebra, cycle_zebra) + L1(horse, cycle_horse)

            # identity losses
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_loss = L1(zebra, identity_zebra) + L1(horse, identity_horse)

            G_loss = (
                loss_G_H + loss_G_Z +
                config.LAMBDA_CYCLE * cycle_loss +
                config.LAMBDA_IDENTITY * identity_loss
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())

        if idx % 50 == 0:
            loop.set_description(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
            loop.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item())
            save_image(horse * 0.5 + 0.5, f"saved_images/real_horse_{epoch}_{idx}.png")
            save_image(zebra * 0.5 + 0.5, f"saved_images/real_zebra_{epoch}_{idx}.png")
            save_image(fake_horse * 0.5 + 0.5, f"saved_images/fake_horse_{epoch}_{idx}.png")
            save_image(fake_zebra * 0.5 + 0.5, f"saved_images/fake_zebra_{epoch}_{idx}.png")


def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    G_losses = []
    D_losses = []
    os.makedirs("loss_plots", exist_ok=True)

    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    opt_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.99)
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR + '/horses',
        root_zebra=config.TRAIN_DIR + '/zebras',
        transform=config.transforms
        
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )

    # ✅ CPU-safe scalers
    g_scaler = GradScaler()
    d_scaler = GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE)

    for epoch in range(config.NUM_EPOCHS):
        train(disc_H, disc_Z, gen_Z, gen_H, loader, opt_gen, opt_disc, L1, mse, d_scaler, g_scaler, epoch,G_losses,D_losses)

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)
    pd.DataFrame({
        "Generator Loss": G_losses,
        "Discriminator Loss": D_losses
    }).to_csv("loss_plots/cyclegan_losses.csv", index=False)
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("CycleGAN Training Losses")
    plt.show()
    plt.savefig("loss_plots/cyclegan_losses.png")
    plt.close()


if __name__ == "__main__":
    main()
